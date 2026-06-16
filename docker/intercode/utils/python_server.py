import rpyc
import sys
from io import StringIO
ORIGINAL_GLOBAL = dict(globals())
class MyService(rpyc.Service):
    def __init__(self):
        self.globals = dict(ORIGINAL_GLOBAL)
    def on_connect(self, conn):
        pass
    def on_disconnect(self, conn):
        pass
    def exposed_execute(self, command):
        try:
            if command == "RESET_CONTAINER_SPECIAL_KEYWORD":
                self.globals = dict(ORIGINAL_GLOBAL)
            output_buffer = StringIO()
            error_buffer = StringIO()
            sys.stdout = output_buffer
            sys.stderr = error_buffer
            exec(command, self.globals)
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            output = output_buffer.getvalue().strip()
            error = error_buffer.getvalue().strip()
            return {
                "output": output,
                "error": error
            }
        except Exception as e:
            return {"error": f"Error: {e}"}
if __name__ == "__main__":
    from rpyc.utils.server import ThreadPoolServer
    server = ThreadPoolServer(MyService(), port=3006)
    server.start()
