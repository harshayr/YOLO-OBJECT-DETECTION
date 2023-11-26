import sys
import os

def error_detail(error , error_details:sys):
    _,_,exc_tb = error_details.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    error_msg = f"Error occur in python script {0} line number {1} error msg {2}".format(filename,exc_tb.tb_lineno,error)
    return error_msg

class CustomException(Exception):
    def __init__(self,error, error_details:sys) :
        super().__init__()
        self.error = error
        self.error_detail = error_details
    def raise_error(self):
        raise error_detail(self.error,self.error_detail)

