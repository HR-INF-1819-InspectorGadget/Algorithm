# Import required libraries
import socket as system_information
import pypyodbc as sql_server
import igCheckModule


# Create a class that will be used to manage connections with the database
class Database:
    # Create a constructor that's able to take in the required information
    def __init__(self, database_name: str, username: str = '', password: str = '', server_name: str = system_information.gethostname(), driver: str = "SQL Server", trusted_connection: str = "Yes"):
        # Check whether all provided values are of the right type and have content
        if igCheckModule.VariableChecks.check_multiple_values_for_type_and_minimal_count([database_name, server_name, driver, trusted_connection], str, 1) and isinstance(username, str) and isinstance(password, str):
            # Set the class variables
            self.database_name = database_name
            self.username = username
            self.password = password
            self.server_name = server_name
            self.driver = driver
            self.trusted_connection = trusted_connection

            # Test the connection and save the result
            connection_test_result = self.test_database_connection()

            # Check whether the test failed
            if not connection_test_result:
                raise ConnectionError("Could not connect to the database")
        else:
            raise ValueError("Please enter information when trying to create an instance of the class")

    # Create a method that tests the connection to the database
    def test_database_connection(self) -> bool:
        # Start the attempt to create a connection
        try:
            # Open a connection to the database
            database_connection = sql_server.connect(self.get_connection_string())

            # Close the connection with the database
            database_connection.close()

            # The connection has opened successfully, return true
            return True
        except:
            # Something went wrong, return false
            return False

    # Create a method that returns the connection string
    def get_connection_string(self) -> str:
        # Return the connection string
        return 'Driver={' + self.driver + '};Server=' + self.server_name + ';Database=' + self.database_name + ';Trusted_Connection=' + self.trusted_connection + ';uid=' + self.username + ';pwd=' + self.password + ''

    # Create a method that runs a query on the database and return the results as an array
    def get_data_from_the_database(self, query: str) -> list:
        # Check whether the query is a string of the minimal length
        if isinstance(query, str) and len(query) >= 15:
            # Check whether the query starts with a select statement and end with a ; of which only 1 is present
            if query[0:6].lower() == "select" and query[len(query) - 1] == ";" and query.count(';') == 1:
                # Start a connection with the database
                database_connection = sql_server.connect(self.get_connection_string())

                # Create a cursor that will execute the query
                cursor = database_connection.cursor()

                # Enter the attempt of receiving information from the server
                try:
                    # Execute the query
                    cursor.execute(query)

                    # Save the results of the query
                    query_results = cursor.fetchall()

                    # Close the connection with the database
                    database_connection.close()

                    # Return the query result
                    return query_results
                except:
                    # Something went wrong
                    return []
            else:
                # The format in which the string has been received is incorrect
                raise SyntaxError("The received query is not of the SELECT type, is not a single query or is not being closed correctly")
        else:
            # The value we received is incompatible with the function
            raise ValueError("The query used to get data from the database should be at least 15 characters")
