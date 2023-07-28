import os
import sys

# Get the current directory
current_dir = os.getcwd()

# Get the script's file name
script_file = os.path.basename(sys.argv[0])

# Get a list of all files in the current directory
file_list = os.listdir(current_dir)

# Filter out subdirectories and keep only Python files
python_files = [
    file for file in file_list if os.path.isfile(file) and file.endswith(".py")
]

# Get additional filenames passed as arguments
additional_files = sys.argv[1:]

# Open the output file
with open("repository_summary.txt", "w") as output_file:
    # Iterate over the Python files
    for file in python_files:
        # Skip the script file and additional filenames
        if file == script_file or file in additional_files:
            continue

        # Write the filename to the output file
        output_file.write(file + "\n")
        output_file.write("```\n")

        # Read the content of the Python file
        with open(file, "r") as input_file:
            code = input_file.read()

        # Write the code to the output file
        output_file.write(code)
        output_file.write("```\n")
