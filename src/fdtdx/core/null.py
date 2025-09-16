
class Null:
    
    __slots__ = ()
    
    def __repr__(self) -> str:
        return "null"
    
    def __str__(self) -> str:
        return repr(self)
    
    __bool__ = lambda _: False


NULL = Null()