def generateCode(classes, inheritance):
    actualClasses = []
    for classObject in classes:
        actualClass = {"name": "", "interface": False, "abstract": False, "inheritsFrom": {}, "attributes": [],
                       "methods": []}
        if "<<abstract>>" in classObject["ClassName"]:
            actualClass["abstract"] = True
            actualClass["name"] = classObject["ClassName"].replace("<<abstract>>", "")
        elif "<<interface>>" in classObject["ClassName"]:
            actualClass["interface"] = True
            actualClass["name"] = classObject["ClassName"].replace("<<interface>>", "")
        else:
            actualClass["name"] = classObject["ClassName"]

        attributes = []
        for attribute in classObject["Attributes"]:
            actualAttribute = {"name": "", "public": False, "type": ""}
            actualAttribute["public"] = attribute[0] == "+"
            actualAttribute["name"] = attribute[2:attribute.find(":")]
            actualAttribute["type"] = attribute[attribute.find(":") + 2:]
            attributes.append(actualAttribute)
        actualClass["attributes"] = attributes

        methods = []
        for method in classObject["Methods"]:
            actualMethod = {"name": "", "public": False, "arguments": [], "returnType": ""}
            actualMethod["public"] = method[0] == "+"
            actualMethod["name"] = method[2:method.find("(")]
            if ":" in method:
                actualMethod["returnType"] = method[method.find(":") + 1:]
            else:
                actualMethod["returnType"] = "null"
            if "(" in method and ")" in method:
                actualMethod["arguments"] = (method[method.find("(") + 1:method.find(")")]).split(", ")
            else:
                actualMethod["arguments"] = []
            methods.append(actualMethod)
        actualClass["methods"] = methods
        actualClasses.append(actualClass)

    for (child, parent) in inheritance:
        child = child.replace("<<abstract>>", "").replace("<<interface>>", "").replace("\n", "")
        parent = parent.replace("<<abstract>>", "").replace("<<interface>>", "").replace("\n", "")
        for actualClass in actualClasses:
            if actualClass["name"] == child:
                actualClass["inheritsFrom"] = parent

    print(actualClasses)

    javaCodes = []
    for actualClass in actualClasses:
        javaCode = json_to_java_class(actualClass)
        print(javaCode)
        javaCodes.append(javaCode)
        write_java_class_to_file(javaCode, actualClass["name"])


def json_to_java_class(json_obj):
    # Ensure the input is a dictionary
    if not isinstance(json_obj, dict):
        raise ValueError("Input must be a dictionary")

    # Get class name
    class_name = json_obj.get('name', 'UnknownClass')

    # Initialize the Java class code
    java_class_code = f'public class {class_name} '

    # Check if the class inherits from another class
    inherits_from = json_obj.get('inheritsFrom', None)
    if inherits_from:
        java_class_code += f'extends {inherits_from} '

    java_class_code += '{\n'

    # Process attributes
    attributes = json_obj.get('attributes', [])
    for attribute in attributes:
        attr_name = attribute.get('name', 'unknownAttribute')
        attr_type = attribute.get('type', 'Object')
        java_class_code += f'    private {attr_type} {attr_name};\n'

    java_class_code += '\n'

    # Process methods
    methods = json_obj.get('methods', [])
    for method in methods:
        method_name = method.get('name', 'unknownMethod')
        return_type = method.get('returnType', 'void')
        arguments = ', '.join(method.get('arguments', []))
        java_class_code += f'    public {return_type} {method_name}({arguments})' + ' {\n'
        java_class_code += '        // Your method implementation here\n'
        java_class_code += '    }\n\n'

    java_class_code += '}'

    return java_class_code


def write_java_class_to_file(java_class_code, file_name):
    with open(file_name, 'w') as java_file:
        java_file.write(java_class_code)
