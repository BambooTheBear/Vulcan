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
        elif "«abstract»" in classObject["ClassName"]:
            actualClass["abstract"] = True
            actualClass["name"] = classObject["ClassName"].replace("«abstract»", "")
        elif "«interface»" in classObject["ClassName"]:
            actualClass["interface"] = True
            actualClass["name"] = classObject["ClassName"].replace("«interface»", "")
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
        javaCode = json_to_java_class(actualClass, actualClasses)
        print(javaCode)
        javaCodes.append(javaCode)
        write_java_class_to_file(javaCode, actualClass["name"]+".java")


def json_to_java_class(class_info, classes):
    # Initialize a variable to store the generated Java code
    java_code = []

    # Helper function to generate attribute declarations
    def generate_attributes(attributes):
        attribute_declarations = []
        for attr in attributes:
            access_modifier = 'public' if attr['public'] else 'private'
            attribute_declarations.append(f'{access_modifier} {attr["type"]} {attr["name"]};')
        return attribute_declarations

    # Helper function to generate method declarations
    def generate_methods(methods, is_interface):
        method_declarations = []
        for method in methods:
            access_modifier = 'public' if method['public'] else 'private'
            return_type = 'void' if method['returnType'] == 'null' else method['returnType']
            method_name = method['name']
            arguments = ', '.join(method['arguments'])

            if is_interface:
                method_declarations.append(f'{access_modifier} abstract {return_type} {method_name}({arguments});')
            else:
                method_declarations.append(f'{access_modifier} {return_type} {method_name}({arguments}) {{ \n \t \t //Implementation \n \t}}')

        return method_declarations

    # Iterate through JSON objects and generate Java code
    class_name = class_info['name']
    is_interface = class_info['interface']
    is_abstract = class_info['abstract']
    inherits_from = class_info['inheritsFrom']
    attributes = class_info['attributes']
    methods = class_info['methods']

    # Build class declaration
    class_declaration = "public "
    if is_interface:
        class_declaration += 'interface '
    else:
        if is_abstract:
            class_declaration += 'abstract '
        class_declaration += 'class '

    class_declaration += class_name

    # Check if the class inherits from another class or interface
    if inherits_from != {}:
        for parent in classes:
            if parent['name'] == inherits_from:
                parent_type = 'implements' if is_interface else 'extends'
                class_declaration += f' {parent_type} {inherits_from}'

    class_declaration += ' {'

    # Generate attribute declarations
    attribute_declarations = generate_attributes(attributes)

    # Generate method declarations
    method_declarations = generate_methods(methods, is_interface)

    print(class_declaration)
    print(method_declarations)
    print(attribute_declarations)

    # Add class declaration, attributes, and methods to the Java code
    java_code.extend([class_declaration])
    java_code.extend(['    ' + attr_decl for attr_decl in attribute_declarations])
    java_code.extend(['    ' + method_decl for method_decl in method_declarations])

    java_code.append('}')

    # Join the generated lines into a single string
    return '\n'.join(java_code)


def write_java_class_to_file(java_class_code, file_name):
    with open(file_name, 'w') as java_file:
        java_file.write(java_class_code)
