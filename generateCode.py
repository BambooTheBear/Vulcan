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
