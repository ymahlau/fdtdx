class Null:
    __slots__ = ()

    def __repr__(self) -> str:
        return "null"

    def __str__(self) -> str:
        return repr(self)

    def __bool__(self):
        return False


NULL = Null()
