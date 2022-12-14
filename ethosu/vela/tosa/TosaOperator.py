# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tosa

import flatbuffers

class TosaOperator(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsTosaOperator(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = TosaOperator()
        x.Init(buf, n + offset)
        return x

    # TosaOperator
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # TosaOperator
    def Op(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

    # TosaOperator
    def AttributeType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, o + self._tab.Pos)
        return 0

    # TosaOperator
    def Attribute(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            from flatbuffers.table import Table
            obj = Table(bytearray(), 0)
            self._tab.Union(obj, o)
            return obj
        return None

    # TosaOperator
    def Inputs(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.String(a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return ""

    # TosaOperator
    def InputsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # TosaOperator
    def Outputs(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.String(a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return ""

    # TosaOperator
    def OutputsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # TosaOperator
    def QuantInfoType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, o + self._tab.Pos)
        return 0

    # TosaOperator
    def QuantInfo(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            from flatbuffers.table import Table
            obj = Table(bytearray(), 0)
            self._tab.Union(obj, o)
            return obj
        return None

def TosaOperatorStart(builder): builder.StartObject(7)
def TosaOperatorAddOp(builder, op): builder.PrependUint32Slot(0, op, 0)
def TosaOperatorAddAttributeType(builder, attributeType): builder.PrependUint8Slot(1, attributeType, 0)
def TosaOperatorAddAttribute(builder, attribute): builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(attribute), 0)
def TosaOperatorAddInputs(builder, inputs): builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(inputs), 0)
def TosaOperatorStartInputsVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def TosaOperatorAddOutputs(builder, outputs): builder.PrependUOffsetTRelativeSlot(4, flatbuffers.number_types.UOffsetTFlags.py_type(outputs), 0)
def TosaOperatorStartOutputsVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def TosaOperatorAddQuantInfoType(builder, quantInfoType): builder.PrependUint8Slot(5, quantInfoType, 0)
def TosaOperatorAddQuantInfo(builder, quantInfo): builder.PrependUOffsetTRelativeSlot(6, flatbuffers.number_types.UOffsetTFlags.py_type(quantInfo), 0)
def TosaOperatorEnd(builder): return builder.EndObject()
