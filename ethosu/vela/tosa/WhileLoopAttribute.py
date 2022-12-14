# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tosa

import flatbuffers

class WhileLoopAttribute(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsWhileLoopAttribute(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = WhileLoopAttribute()
        x.Init(buf, n + offset)
        return x

    # WhileLoopAttribute
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # WhileLoopAttribute
    def CondBranch(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # WhileLoopAttribute
    def BodyBranch(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

def WhileLoopAttributeStart(builder): builder.StartObject(2)
def WhileLoopAttributeAddCondBranch(builder, condBranch): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(condBranch), 0)
def WhileLoopAttributeAddBodyBranch(builder, bodyBranch): builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(bodyBranch), 0)
def WhileLoopAttributeEnd(builder): return builder.EndObject()
