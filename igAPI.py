# Import the packages required to run the API
import http.server
import socketserver
import igDatabaseModule


# Create a custom version of the request handler that allows us to create a custom implementation of available requests
# noinspection PyShadowingBuiltins
class IGRequestHandler(http.server.SimpleHTTPRequestHandler):
    # Override the log method just to disable it
    def log_message(self, format, *args):
        pass

    # Override the do_get method in order to define a custom get
    def do_GET(self):
        # We received a request, log it
        print("[c] <- Receiving data from " + self.client_address[0] + " requesting: " + self.path)

        # Create an instance of the database
        database = igDatabaseModule.Database("PGA_HRO")

        # Send the response status code
        self.send_response(200)

        # Send the headers telling the client about the data
        self.send_header('Content-type', 'text/html')
        self.end_headers()

        # Check the type of request we received
        if self.path == '/test_connection':
            # Set the title for the response
            response_title = "Connection Test Response"

            # Tell the user that the connection to the API was successful
            response_message = "API Connection: Success<br/>"

            # Check whether we have a solid connection to the database
            if database.test_database_connection():
                # The connection was successful, tell the user
                response_message = response_message + "Database Connection: Success"
            else:
                # The connection was unsuccessful, tell the user
                response_message = response_message + "Database Connection: Failed"
        else:
            # Set the title for the response
            response_title = "Address Not Found Response"

            # We didn't receive a good request, return an empty string
            response_message = 'Something'

        # Log the response
        print("[c] -> Returning: " + response_title)

        # Write the content as data
        self.wfile.write(bytes(response_message, "utf8"))

        # Return the data
        return


# Define the port that we will use to listen to our API
api_port = 8080

# Create an instance for a request handler
http_request_handler = IGRequestHandler

# Create a variable that will store information about the http connection
http_connection = socketserver.TCPServer(("127.0.0.1", api_port), http_request_handler)

# Let the console application know how the API is doing
print("[-] API Starting\n[v] API Started\n[i] Listening to port: " + str(api_port) + " at address: " + str(http_connection.server_address))

# Attempt to start a connection
try:
    # Start the connection
    http_connection.serve_forever()
except:
    # Something went wrong, tell the user
    print("[X] The startup of the API was interrupted... please try again later")

    # Stop the application
    quit(1)
