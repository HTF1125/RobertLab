"""ROBERT"""

class DataStore(dict):
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value

    def flatten(self):
        """
        Flatten the nested data structure into a single-level dictionary.
        Returns:
            dict: A flattened dictionary.
        """
        flattened = {}

        def _flatten(dictionary, prefix=""):
            for key, value in dictionary.items():
                if isinstance(value, DataStore):
                    _flatten(value, prefix + key + ".")
                else:
                    flattened[prefix + key] = value

        _flatten(self)
        return flattened

    def load(self, data):
        """
        Load data into the DataStore.
        Args:
            data (dict): The data to be loaded.
        """
        for key, value in data.items():
            if isinstance(value, dict):
                self[key] = DataStore()
                self[key].load(value)
            else:
                self[key] = value

    def to_dict(self):
        """
        Convert the DataStore into a regular nested dictionary.
        Returns:
            dict: The nested dictionary representing the DataStore.
        """
        result = {}
        for key, value in self.items():
            if isinstance(value, DataStore):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
