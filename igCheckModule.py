# Import the required packages
from typing import Union
from typing import List


# Create a class that will store all methods checking values
class VariableChecks:
    @staticmethod
    def check_type_and_minimal_count(value: Union[str, int, list, float], expected_type: type, count: int) -> bool:
        # Check whether count contains a positive integer
        if count >= 0:
            # Check whether the type of the value matches
            if isinstance(value, expected_type):
                # Check whether the type is an integer or a float
                if isinstance(value, int) or isinstance(value, float):
                    # Check whether the integer matches the count
                    if value >= count:
                        # The value matches the count
                        return True
                elif isinstance(value, str) or isinstance(value, list):
                    # Check whether the length of the variables matches the count
                    if len(value) >= count:
                        # The length of the value is correct
                        return True
                else:
                    # We've received a type this function is not build to handle
                    raise ValueError("The value you are trying to check cannot be analysed for length. Please use isinstance() instead")
            # We haven't been able to return true, which implies that one of the checks failed
            return False
        else:
            # We cannot have a negative count
            raise ValueError("The value cannot be checked for a negative count")

    @staticmethod
    def check_multiple_values_for_type_and_minimal_count(values: List[Union[str, int, list, float]], expected_type: type, count: int) -> bool:
        # Check whether the list is not empty
        if len(values) > 0:
            # Loop over the items in the list
            for value in values:
                # Check the information of the current value and save the result
                result = VariableChecks.check_type_and_minimal_count(value, expected_type, count)

                # Check whether we received a false result
                if not result:
                    # One failing, means everything fails
                    return False

            # Nothing has failed, the test has been a success
            return True
        else:
            # We received a bad request
            raise ValueError("In order to check multiple values for type and count, we need a list with content")
