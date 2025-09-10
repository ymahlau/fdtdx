# Unitful design decisions


* A unitful can have the following values:
    * any python internal datatypes (including int and bool, which only work without a unit)
    * np.ndarray and np.number
    * jax Array (traced or not-traced)

* Unitfuls try to keep the data type consistent whenever possible:
    * python scalars stay python values if possible. if not, they are converted to jax arrays.
    * numpy arrays always stay numpy arrays -> TODO!!!!
    * operations on unitfuls with numpy / jax arrays mimic the standard numpy / jax behavior

* For traced arrays, the actual static value is carried along the computation if:
    * array is not too large
    * computation can be performed (e.g. multiplication with standard traced jax array breaks information flow)
    * if global stop flag is not set

* We try to prevent these breaks in information flow through replacing standard jax arrays with unitfuls (with empty units) during tracing
    * only happens if array is not too large
    * only happens within jit context, when leaving jit context the unitfuls are materialised
    * only if global stop flag is not set

* The scale of a unitful is optimized
    * if it is possible to do so (proper value type, not traced, etc...)


