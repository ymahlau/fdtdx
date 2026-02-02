import pytest

from fdtdx.core.null import NULL, Null


def test_null_repr():
    """Test that Null.__repr__ returns 'null'"""
    null_obj = Null()
    assert repr(null_obj) == "null"


def test_null_str():
    """Test that Null.__str__ returns 'null'"""
    null_obj = Null()
    assert str(null_obj) == "null"


def test_null_bool_is_false():
    """Test that Null evaluates to False in boolean context"""
    null_obj = Null()
    assert bool(null_obj) is False
    assert not null_obj


def test_null_in_conditional():
    """Test that Null behaves correctly in if statements"""
    null_obj = Null()
    if null_obj:
        pytest.fail("Null should evaluate to False in conditional")


def test_null_singleton_exists():
    """Test that NULL singleton is an instance of Null"""
    assert isinstance(NULL, Null)


def test_null_singleton_repr():
    """Test that NULL singleton repr returns 'null'"""
    assert repr(NULL) == "null"


def test_null_singleton_str():
    """Test that NULL singleton str returns 'null'"""
    assert str(NULL) == "null"


def test_null_singleton_bool():
    """Test that NULL singleton evaluates to False"""
    assert bool(NULL) is False
    assert not NULL


def test_null_has_slots():
    """Test that Null class uses __slots__ for memory efficiency"""
    assert hasattr(Null, "__slots__")
    assert Null.__slots__ == ()


def test_null_no_dict():
    """Test that Null instances don't have __dict__ due to __slots__"""
    null_obj = Null()
    assert not hasattr(null_obj, "__dict__")


def test_null_cannot_add_attributes():
    """Test that Null instances cannot have attributes added due to __slots__"""
    null_obj = Null()
    with pytest.raises(AttributeError):
        null_obj.some_attribute = "value"
