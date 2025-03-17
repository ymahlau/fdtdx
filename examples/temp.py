
import jax
import jax.numpy as jnp
from fdtdx.core.jax.pytrees import ExtendedTreeClass, extended_autoinit


@extended_autoinit
class A(ExtendedTreeClass):
    a: float = 1.0
    x: jax.Array = jnp.ones((2, 2), dtype=jnp.float32)

@extended_autoinit
class B(ExtendedTreeClass):
    b: float = 2.0

@extended_autoinit
class C(A, B):
    c: float = 3.0
    


def main():
    c = C()
    print(c)


if __name__ == '__main__':
    main()
