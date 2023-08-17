# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tosa

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class ClampAttribute(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ClampAttribute()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsClampAttribute(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def ClampAttributeBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x4F\x53\x41", size_prefixed=size_prefixed)

    # ClampAttribute
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # ClampAttribute
    def MinInt(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # ClampAttribute
    def MaxInt(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # ClampAttribute
    def MinFp(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
        return 0

    # ClampAttribute
    def MinFpAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint8Flags, o)
        return 0

    # ClampAttribute
    def MinFpLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # ClampAttribute
    def MinFpIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        return o == 0

    # ClampAttribute
    def MaxFp(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
        return 0

    # ClampAttribute
    def MaxFpAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint8Flags, o)
        return 0

    # ClampAttribute
    def MaxFpLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # ClampAttribute
    def MaxFpIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        return o == 0

def ClampAttributeStart(builder):
    builder.StartObject(4)

def Start(builder):
    ClampAttributeStart(builder)

def ClampAttributeAddMinInt(builder, minInt):
    builder.PrependInt32Slot(0, minInt, 0)

def AddMinInt(builder, minInt):
    ClampAttributeAddMinInt(builder, minInt)

def ClampAttributeAddMaxInt(builder, maxInt):
    builder.PrependInt32Slot(1, maxInt, 0)

def AddMaxInt(builder, maxInt):
    ClampAttributeAddMaxInt(builder, maxInt)

def ClampAttributeAddMinFp(builder, minFp):
    builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(minFp), 0)

def AddMinFp(builder, minFp):
    ClampAttributeAddMinFp(builder, minFp)

def ClampAttributeStartMinFpVector(builder, numElems):
    return builder.StartVector(1, numElems, 1)

def StartMinFpVector(builder, numElems: int) -> int:
    return ClampAttributeStartMinFpVector(builder, numElems)

def ClampAttributeAddMaxFp(builder, maxFp):
    builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(maxFp), 0)

def AddMaxFp(builder, maxFp):
    ClampAttributeAddMaxFp(builder, maxFp)

def ClampAttributeStartMaxFpVector(builder, numElems):
    return builder.StartVector(1, numElems, 1)

def StartMaxFpVector(builder, numElems: int) -> int:
    return ClampAttributeStartMaxFpVector(builder, numElems)

def ClampAttributeEnd(builder):
    return builder.EndObject()

def End(builder):
    return ClampAttributeEnd(builder)
