# SPDX-FileCopyrightText: Copyright 2020-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Description:
# Functions used to write to a TensorFlow Lite format file. Supports adding in file identifiers.
import flatbuffers
import flatbuffers.number_types as N
import numpy as np
from flatbuffers import encode
from flatbuffers.builder import UOffsetTFlags

from ._version import __version__
from .errors import VelaError
from .nn_graph import PassPlacement
from .operation import Op
from .reader_util import align_inputs_indices
from .tensor import MemType
from .tensor import shape_num_elements
from .tensor import TensorPurpose
from .tflite import Buffer
from .tflite import Metadata
from .tflite import Model
from .tflite import Operator
from .tflite import OperatorCode
from .tflite import QuantizationParameters
from .tflite import SubGraph
from .tflite import Tensor
from .tflite_mapping import builtin_operator_inv_map
from .tflite_mapping import BuiltinOperator
from .tflite_mapping import datatype_inv_map
from .tflite_mapping import optype_to_builtintype

# the python flatbuffer interface is missing a method to add in file identifier. patching it in here:

tflite_version = 3
tflite_file_identifier = "TFL" + str(tflite_version)


def FinishWithFileIdentifier(self, rootTable, fid):
    if fid is None or len(fid) != 4:
        raise VelaError("FileIdentifier must be 4 chars")

    flags = N.Uint8Flags
    prepSize = 4
    self.Prep(self.minalign, prepSize + len(fid))
    for i in range(3, -1, -1):
        self.head = self.head - flags.bytewidth
        encode.Write(flags.packer_type, self.Bytes, self.Head(), ord(fid[i]))

    return self.Finish(rootTable)


flatbuffers.Builder.FinishWithFileIdentifier = FinishWithFileIdentifier


def make_vector(v):
    try:
        len(v)
        return v
    except TypeError:
        return [v]


class TFLiteSerialiser:

    # The 0th buffer is always by default an empty buffer that can be used by tensors
    # without any constant data
    BUF_IDX_ZERO = 0
    BUF_IDX_START = 1

    def __init__(self, nng):
        self.builder = flatbuffers.Builder(0)
        self.nng = nng

        self.buf_idx = TFLiteSerialiser.BUF_IDX_START
        self.buffers_to_write = []  # have an empty array there
        self.tensor_map_all = []  # Keep track of all subgraphs
        self.tensor_map_sg = []  # Keep track of one subgraph

        self.ops_to_ignore = (Op.Const, Op.Placeholder, Op.SubgraphInput)

        self.subgraphs_to_write = [sg for sg in self.nng.subgraphs if sg.placement == PassPlacement.Cpu]

        all_ops = []
        for sg in self.subgraphs_to_write:
            for ps in sg.passes:
                for op in ps.ops:
                    if op.type not in self.ops_to_ignore:
                        # swap from nng input indexing to TensorFlow Lite input indexing
                        self.align_nng_inputs_to_tflite(op)
                        all_ops.append(op)
                    if op.type.is_conv2d_op() or op.type.is_depthwise_conv2d_op() or op.type == Op.FullyConnected:
                        # Op is run on CPU, make sure the original weight and bias tensors are written back
                        # instead of the cloned/reshaped (see tflite_reader)
                        # Do nothing when values are None (dynamic weights)
                        if op.inputs[1].values is not None:
                            for idx, inp in enumerate(op.inputs):
                                if inp != op.ifm and inp is not None and inp.src_tensor is not None:
                                    op.inputs[idx] = inp.src_tensor

        # list of tuple(Op, string, op.version); the custom code is only used for 3rd party custom operators
        self.operator_codes = sorted(set((op.type, op.attrs.get("custom_code", ""), op.version) for op in all_ops))
        self.operator_code_map = {}

    def align_nng_inputs_to_tflite(self, op):
        from_indices = op.type.info.indices
        _, _, to_indices = builtin_operator_inv_map[op.type]
        op.inputs = align_inputs_indices(from_indices, to_indices, op.inputs)

    def write_byte_vector(self, v, alignment=1):
        builder = self.builder
        builder.StartVector(1, len(v), alignment)
        for e in v[::-1]:
            builder.PrependByte(e)
        return builder.EndVector()

    def write_int_vector(self, v):
        builder = self.builder
        builder.StartVector(4, len(v), 4)
        for e in v[::-1]:
            builder.PrependInt32(e)
        return builder.EndVector()

    def write_long_vector(self, v):
        builder = self.builder
        builder.StartVector(8, len(v), 8)
        for e in v[::-1]:
            builder.PrependInt64(e)
        return builder.EndVector()

    def write_float_vector(self, v):
        builder = self.builder
        builder.StartVector(4, len(v), 4)
        for e in v[::-1]:
            builder.PrependFloat32(e)
        return builder.EndVector()

    def write_offset_vector(self, v):
        builder = self.builder
        builder.StartVector(4, len(v), 4)
        for e in v[::-1]:
            builder.PrependUOffsetTRelative(e)
        return builder.EndVector()

    def assign_buffers_to_tensors(self, tensors, scratch_tensor):
        if scratch_tensor is not None:
            scratch_tensor_mem_area = scratch_tensor.mem_area
        else:
            scratch_tensor_mem_area = None  # all tensors are initialised to MemArea.Unknown

        buffer_map = {}

        for tens in tensors:
            # Set buffer ids depending on allocation
            if tens.is_allocated_in_tensor_arena(scratch_tensor_mem_area) or tens.mem_type == MemType.Scratch_fast:
                # Tensor allocated in the scratch areas, does not have any constant data and can
                # therefore all point to the empty buffer (zero)
                buffer_map[tens] = TFLiteSerialiser.BUF_IDX_ZERO
            else:
                buffer_map[tens] = self.buf_idx
                self.buf_idx += 1

        # Initialize/extend buffers_to_write to a length equal to number of buffers so
        # they can be appended at the correct index during tensor serialization
        self.buffers_to_write += [None] * (self.buf_idx)

        return buffer_map

    def serialise_operator_code(self, idx, op_type, custom_code, version):
        builder = self.builder
        custom_code_offset = None
        if op_type == Op.Custom:
            tf_code, opt_serializer, _ = builtin_operator_inv_map[op_type]
            custom_code_offset = builder.CreateString(custom_code)
        else:
            assert (
                op_type in builtin_operator_inv_map
            ), "Vela does not contain a mapping to serialise {} operator to a TensorFlow Lite operator".format(op_type)
            tf_code, opt_serializer, _ = builtin_operator_inv_map[op_type]

            if op_type == Op.CustomNpuOp:
                assert (
                    tf_code == BuiltinOperator.CUSTOM
                ), "Vela only supports serialising NpuOp operators as TensorFlow Lite Custom operators"
                custom_code_offset = builder.CreateString("ethos-u")

        # there can be multiple different types of 3rd party custom operators (i.e. non-"ethos-u" ones). therefore we
        # need to add an extra level of indirection to this particular entry in the operator_code_map to allow for the
        # correct lookup later on
        if op_type == Op.Custom:
            if op_type not in self.operator_code_map:
                self.operator_code_map[op_type] = {}
            self.operator_code_map[op_type][custom_code] = (idx, tf_code, opt_serializer)
        else:
            self.operator_code_map[op_type] = (idx, tf_code, opt_serializer)

        OperatorCode.OperatorCodeStart(builder)
        OperatorCode.OperatorCodeAddDeprecatedBuiltinCode(builder, tf_code if tf_code < 127 else 127)
        OperatorCode.OperatorCodeAddBuiltinCode(builder, tf_code)
        OperatorCode.OperatorCodeAddVersion(builder, version)
        if custom_code_offset is not None:
            OperatorCode.OperatorCodeAddCustomCode(builder, custom_code_offset)

        return OperatorCode.OperatorCodeEnd(builder)

    def serialise_quantization_parameters(self, quant):
        builder = self.builder

        qp = None
        min = None
        max = None
        scale = None
        zero_point = None
        if quant is not None:
            if quant.min is not None:
                min = self.write_float_vector(make_vector(quant.min))
            if quant.max is not None:
                max = self.write_float_vector(make_vector(quant.max))
            if quant.scale_f32 is not None:
                scale = self.write_float_vector(make_vector(quant.scale_f32))
            if quant.zero_point is not None:
                zero_point = self.write_long_vector(make_vector(quant.zero_point))

            QuantizationParameters.QuantizationParametersStart(builder)
            if min is not None:
                QuantizationParameters.QuantizationParametersAddMin(builder, min)
            if max is not None:
                QuantizationParameters.QuantizationParametersAddMax(builder, max)
            if scale is not None:
                QuantizationParameters.QuantizationParametersAddScale(builder, scale)
            if zero_point is not None:
                QuantizationParameters.QuantizationParametersAddZeroPoint(builder, zero_point)
            if quant.quant_dim is not None:
                QuantizationParameters.QuantizationParametersAddQuantizedDimension(builder, quant.quant_dim)
            qp = QuantizationParameters.QuantizationParametersEnd(builder)

        return qp

    def serialise_tensor(self, tens):
        builder = self.builder
        if shape_num_elements(tens.original_shape) != shape_num_elements(tens.shape):
            # shapes have changed size, therefore assume that the latest (modified) shape is correct
            tens_shape = tens.shape
        else:
            # shapes have not changed size, therefore the original shape is valid
            tens_shape = tens.original_shape
        values = tens.values

        buf_id = self.buffer_map[tens]
        # Sanity check that if buffer 0 is used there must not be any data
        assert not (buf_id == TFLiteSerialiser.BUF_IDX_ZERO and values is not None)
        self.buffers_to_write[buf_id] = None if values is None else values.flatten().view(np.uint8)

        shape = self.write_int_vector(tens_shape)

        name = builder.CreateString(tens.name)
        quant = self.serialise_quantization_parameters(tens.quantization)

        Tensor.TensorStart(builder)
        Tensor.TensorAddShape(builder, shape)
        Tensor.TensorAddType(builder, datatype_inv_map[tens.dtype])
        # All tensors must have a valid backing buffer, even if it is empty.
        # Empty buffers should be kept unique for TensorFlow Lite Micro
        Tensor.TensorAddBuffer(builder, buf_id)
        Tensor.TensorAddName(builder, name)
        if quant is not None:
            Tensor.TensorAddQuantization(builder, quant)
        Tensor.TensorAddIsVariable(builder, tens.is_variable)

        res = Tensor.TensorEnd(builder)
        return res

    def serialise_operator(self, op):
        builder = self.builder

        inputs_offset = self.write_int_vector(
            [self.tensor_map_sg[tens] if tens in self.tensor_map_sg else -1 for tens in op.inputs]
        )
        outputs_offset = self.write_int_vector(
            [self.tensor_map_sg[tens] for tens in op.outputs if tens in self.tensor_map_sg]
        )
        intermediates_offset = self.write_int_vector(
            [self.tensor_map_sg[tens] for tens in op.intermediates if tens in self.tensor_map_sg]
        )

        if op.type == Op.Custom:
            op_idx, tflop, opt_serializer = self.operator_code_map[op.type][op.attrs.get("custom_code", "")]
        else:
            op_idx, tflop, opt_serializer = self.operator_code_map[op.type]

        builtin_opt_offset = None
        custom_opt_offset = None
        if opt_serializer is not None:
            attrs = dict(op.attrs)
            if op.run_on_npu:
                if "strides" in attrs:
                    attrs["stride_h"] = attrs["strides"][1]
                    attrs["stride_w"] = attrs["strides"][2]
                if "ksize" in attrs:
                    attrs["filter_height"] = attrs["ksize"][1]
                    attrs["filter_width"] = attrs["ksize"][2]
                if "dilation" in attrs:
                    attrs["dilation_h_factor"] = attrs["dilation"][1]
                    attrs["dilation_w_factor"] = attrs["dilation"][2]
                if "channel_multiplier" in attrs:
                    attrs["depth_multiplier"] = attrs["channel_multiplier"]
                attrs["fused_activation_function"] = op.activation.op_type if op.activation is not None else None

            # Serialize VarHandleOptions (only op that have attributes with type String)
            if "container" in attrs:
                attrs["container"] = builder.CreateString(attrs["container"])
            if "shared_name" in attrs:
                attrs["shared_name"] = builder.CreateString(attrs["shared_name"])

            builtin_opt_offset, custom_opt_offset = opt_serializer.serialize(builder, attrs)

            # report any missing attributes that could not be written during serialize().
            # operators that have been created internally (i.e. not created as part of reading an input network) may not
            # have the write error attribute
            attribute_write_error = attrs.get("attribute_write_error", [])
            if len(attribute_write_error) != 0:
                print(
                    f"Warning: Could not write the following attributes to {optype_to_builtintype(op.type)}"
                    f" '{op.name}' {opt_serializer.name} field: {', '.join(attribute_write_error)}"
                )

        mutating_variable_inputs_offset = self.write_byte_vector([])
        Operator.OperatorStart(builder)
        Operator.OperatorAddOpcodeIndex(builder, op_idx)
        Operator.OperatorAddInputs(builder, inputs_offset)
        Operator.OperatorAddOutputs(builder, outputs_offset)
        Operator.OperatorAddIntermediates(builder, intermediates_offset)

        if builtin_opt_offset is not None:
            Operator.OperatorAddBuiltinOptionsType(builder, opt_serializer.builtin_opt_type)
            Operator.OperatorAddBuiltinOptions(builder, builtin_opt_offset)
        if custom_opt_offset is not None:
            Operator.OperatorAddCustomOptions(builder, custom_opt_offset)
            Operator.OperatorAddCustomOptionsFormat(builder, opt_serializer.custom_opt_format)

        Operator.OperatorAddMutatingVariableInputs(builder, mutating_variable_inputs_offset)
        return Operator.OperatorEnd(builder)

    def serialise_subgraph(self, sg, name):
        builder = self.builder
        all_ops = []
        placeholder_ops = []

        for ps in sg.passes:
            for op in ps.ops:
                if op.type not in self.ops_to_ignore:
                    all_ops.append(op)
                elif op.type == Op.Placeholder:
                    placeholder_ops.append(op)

        # Make sure all original tensors are written back, special case for Ops
        # with connected subgraphs. Even though not all inputs are used,
        # the reference kernel expects all inputs to be in the tflite file.
        # Since we traverse the graph starting with all outputs they are
        # always added but if an input is not referenced it will not be added
        # to an op.
        tensor_set = set(sg.original_inputs)

        # Remove any virtual outputs since they are only used internally when
        # traversing the graph.
        for tens in sg.virtual_outputs:
            tens.ops[0].outputs = []
            if tens in sg.output_tensors:
                sg.output_tensors.remove(tens)

        # Add the tensors from all valid ops, as well as the tensors from placeholder ops
        # This allows us to serialise tensors which arent attached to any specific ops,
        # e.g. due to an empty graph containing no ops
        for op in all_ops + placeholder_ops:
            for tens in op.inputs + op.outputs + op.intermediates:
                if tens is not None:
                    tensor_set.add(tens)

        all_tensors = [tens for nm, idx, tens in sorted((tens.name, idx, tens) for idx, tens in enumerate(tensor_set))]

        scratch_tensors = [tens for tens in all_tensors if tens.purpose is TensorPurpose.Scratch]

        if len(scratch_tensors) == 0:
            scratch_tensor = None
        else:
            assert len(scratch_tensors) == 1, "Multiple scratch tensors"
            scratch_tensor = scratch_tensors[0]

        self.tensor_map_sg = {tens: idx for idx, tens in enumerate(all_tensors)}
        self.buffer_map = self.assign_buffers_to_tensors(all_tensors, scratch_tensor)
        self.tensor_map_all.append(self.tensor_map_sg)

        tensors_offset = self.write_offset_vector([self.serialise_tensor(tens) for tens in all_tensors])

        # Make sure the input_tensors haven't been modified
        assert all(inp in sg.original_inputs for inp in sg.input_tensors)
        inputs = [self.tensor_map_sg[tens] for tens in sg.original_inputs if tens in self.tensor_map_sg]

        inputs_offset = self.write_int_vector(inputs)
        outputs_offset = self.write_int_vector(
            [self.tensor_map_sg[tens] for tens in sg.output_tensors if tens in self.tensor_map_sg]
        )

        operators_offset = self.write_offset_vector([self.serialise_operator(op) for op in all_ops])

        SubGraph.SubGraphStart(builder)
        SubGraph.SubGraphAddTensors(builder, tensors_offset)
        SubGraph.SubGraphAddInputs(builder, inputs_offset)
        SubGraph.SubGraphAddOutputs(builder, outputs_offset)

        SubGraph.SubGraphAddOperators(builder, operators_offset)
        SubGraph.SubGraphAddName(builder, name)

        return SubGraph.SubGraphEnd(builder)

    def write_aligned_bytes(self, buf):
        builder = self.builder
        builder.assertNotNested()
        builder.nested = True
        if isinstance(buf, str):
            data = bytes(buf, "utf-8")
        else:
            data = bytes(buf)
        length_bytes = UOffsetTFlags.py_type(len(data))
        builder.vectorNumElems = length_bytes
        builder.Prep(16, length_bytes)  # Reserve aligned storage
        builder.head = UOffsetTFlags.py_type(builder.Head() - length_bytes)  # Update FlatBuffer internal pointer
        builder.Bytes[builder.Head() : builder.Head() + length_bytes] = data  # Assign bytes to aligned area
        return builder.EndVector()

    def serialise_buffer(self, buf):
        builder = self.builder
        data = None
        if buf is not None:
            data = self.write_aligned_bytes(buf)
        Buffer.BufferStart(builder)
        if data is not None:
            Buffer.BufferAddData(builder, data)
        return Buffer.BufferEnd(builder)

    def serialise_metadata(self, metadata):
        builder = self.builder
        name = builder.CreateString(metadata[0])

        Metadata.MetadataStart(builder)
        Metadata.MetadataAddName(builder, name)
        Metadata.MetadataAddBuffer(builder, metadata[1])

        return Metadata.MetadataEnd(builder)

    def serialise_model(self):
        builder = self.builder
        operator_code_offset = self.write_offset_vector(
            [
                self.serialise_operator_code(idx, optype, code, version)
                for idx, (optype, code, version) in enumerate(self.operator_codes)
            ]
        )

        description = builder.CreateString(f"Vela {__version__} Optimised")
        self.nng.metadata.append(("vela_version", __version__))

        subgraph_offset = self.write_offset_vector(
            [self.serialise_subgraph(sg, builder.CreateString(sg.name)) for sg in self.subgraphs_to_write]
        )

        # Fill the metadata buffer
        version = np.int32(0)
        subgraph_idx = np.int32(len(self.subgraphs_to_write))

        nbr_tensors_all = np.sum([len(tensor_map_sg) for tensor_map_sg in self.tensor_map_all], dtype=np.int32)

        offlineAlloc = [version, subgraph_idx, nbr_tensors_all]

        if not any([name == b"OfflineMemoryAllocation" for name, _ in self.nng.metadata]):
            for tensor_map_sg in self.tensor_map_all:
                nbr_tensors_sg = np.int32(len(tensor_map_sg))
                # An offset of -1 indicates that the tensor will be allocated online by Tensorflow Lite Micro
                offsets = [np.int32(-1)] * nbr_tensors_sg
                # Ensure that the order of the offsets match the order of the tensors
                for tens, idx in tensor_map_sg.items():
                    # Set offsets for tensor allocated in Tensor Arena or in the scratch_fast area
                    if tens.mem_type in (MemType.Scratch, MemType.Scratch_fast):
                        offsets[idx] = np.int32(tens.address) if tens.address is not None else np.int32(0)

                offlineAlloc += offsets

            self.nng.metadata.append(("OfflineMemoryAllocation", np.array(offlineAlloc)))

        metadata_list = []
        for name, buffer in self.nng.metadata:
            self.buffers_to_write.append(buffer)
            metadata_list.append((name, len(self.buffers_to_write) - 1))

        buffers_offset = self.write_offset_vector([self.serialise_buffer(buf) for buf in self.buffers_to_write])
        metadata_offset = self.write_offset_vector([self.serialise_metadata(metadata) for metadata in metadata_list])

        Model.ModelStart(builder)
        Model.ModelAddVersion(builder, tflite_version)
        Model.ModelAddOperatorCodes(builder, operator_code_offset)
        Model.ModelAddSubgraphs(builder, subgraph_offset)
        Model.ModelAddDescription(builder, description)
        Model.ModelAddBuffers(builder, buffers_offset)
        Model.ModelAddMetadata(builder, metadata_offset)
        return Model.ModelEnd(builder)

    def serialise(self):

        model = self.serialise_model()

        self.builder.FinishWithFileIdentifier(model, tflite_file_identifier)

        return self.builder.Output()

    def write(self, filename):
        with open(self.filename, "wb") as f:
            f.write(self.serialised_buf)


def write_tflite(nng, filename):
    writer = TFLiteSerialiser(nng)
    buf = writer.serialise()

    with open(filename, "wb") as f:
        f.write(buf)

def write_tflite_buffer(nng):
    writer = TFLiteSerialiser(nng)
    buf = writer.serialise()
    return buf
