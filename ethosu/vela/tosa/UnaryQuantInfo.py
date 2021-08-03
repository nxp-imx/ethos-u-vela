# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tosa

import flatbuffers

class UnaryQuantInfo(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsUnaryQuantInfo(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = UnaryQuantInfo()
        x.Init(buf, n + offset)
        return x

    # UnaryQuantInfo
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # UnaryQuantInfo
    def InputZp(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # UnaryQuantInfo
    def OutputZp(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

def UnaryQuantInfoStart(builder): builder.StartObject(2)
def UnaryQuantInfoAddInputZp(builder, inputZp): builder.PrependInt32Slot(0, inputZp, 0)
def UnaryQuantInfoAddOutputZp(builder, outputZp): builder.PrependInt32Slot(1, outputZp, 0)
def UnaryQuantInfoEnd(builder): return builder.EndObject()