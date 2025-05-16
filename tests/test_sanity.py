import importlib
import pkgutil

import fdtdx


def test_sanity():
    assert True is True


def test_imports():
    # Dictionary to store successfully imported modules
    imported_modules = {}

    def import_submodules(package, imported_set=None):
        """Recursively import all submodules of package."""
        if imported_set is None:
            imported_set = set()

        if isinstance(package, str):
            package = importlib.import_module(package)

        # Store the fully qualified name of the module
        module_name = package.__name__
        if module_name in imported_set:
            return

        imported_set.add(module_name)
        imported_modules[module_name] = package

        # Check for __path__ attribute which indicates this is a package
        if hasattr(package, "__path__"):
            # Get all submodules
            for _, name, _ in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
                submodule = importlib.import_module(name)
                import_submodules(submodule, imported_set)

        return imported_set

    # Start with the base fdtdx module
    import_submodules(fdtdx)

    # Print summary of imported modules
    print(f"\nSuccessfully imported {len(imported_modules)} modules:")
    for name in sorted(imported_modules.keys()):
        print(f"  - {name}")

    # Verify that at least some modules were imported
    assert len(imported_modules) > 1, "Only the base module was imported, expected submodules"
