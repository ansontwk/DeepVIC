program_header = r"""
██████╗ ███████╗███████╗██████╗ ██╗   ██╗██╗ ██████╗
██╔══██╗██╔════╝██╔════╝██╔══██╗██║   ██║██║██╔════╝
██║  ██║█████╗  █████╗  ██████╔╝██║   ██║██║██║     
██║  ██║██╔══╝  ██╔══╝  ██╔═══╝ ╚██╗ ██╔╝██║██║     
██████╔╝███████╗███████╗██║      ╚████╔╝ ██║╚██████╗
╚═════╝ ╚══════╝╚══════╝╚═╝       ╚═══╝  ╚═╝ ╚═════╝
                                                    
"""
program_desc = "DeepVIC: Deep learning Virulence factor Identifier and Classifier"
program_whatdoesitdo = "Bacterial Virulence factor prediction and classifcation based on protein sequences"
version = "1.0.0"
flavour_text = "'Houston, We have lift-off'"
author = "WK TSUI"

def print_header():
    print(program_header+ "\n")
    print(program_desc)
    print(f"You are currently running version {version}\n")

def print_bye():
    print("Thank you come again.")