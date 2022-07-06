import firelang,sys
VERSION = 1.0
print(f"firelang {VERSION}\n enter quench() to leave interpreter\n\n")
while True:
    try:
        text = input('firelang >>> ')

        if text == "quench()": sys.exit()
        else:
            result, error = firelang.run('<stdin>',text)

            if error: print(error.error_as_string())
            else:
                print(result)
    except KeyboardInterrupt:
        print("Are you trying to leave the interpreter, type exit()\n")