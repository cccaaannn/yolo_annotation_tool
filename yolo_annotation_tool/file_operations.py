def __read_from_file(file_name):
    try:
        with open(file_name,'r', encoding='utf-8') as file:
            content = file.read()
            return content
    except (OSError, IOError) as e:
        print(e)

def __write_to_file(to_write, file_name, write_mode="w"):
    try:
        with open(file_name,write_mode, encoding='utf-8') as file:
            for item in to_write:
                file.write(item.__str__())
                file.write("\n")
    except (OSError, IOError) as e:
        print(e)
