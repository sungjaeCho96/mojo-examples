from common import PointerString

fn str_len(s: PointerString) -> Int:
    var len = 0
    while s[len] != 0:
        len += 1
    return len

fn str_concat(s1: PointerString, s2: PointerString) -> PointerString:
    let l1 = str_len(s1)
    let l2 = str_len(s2)
    let str = PointerString.alloc(l1 + l2 + 1)

    memcpy[UInt8](str, s1, l1)
    memcpy[UInt8](str.offset(l1), s2, l2)
    str.store(l1 + l2, 0)

    return str

fn str_to_ptr(s: String) -> PointerString:
    let ret = PointerString.alloc(len(s) + 1);
    for i in range(len(s)):
        ret.store(i, ord(s[i]))
    ret.store(len(s), 0)
    return ret

fn string_compare(a: PointerString, b: PointerString) -> Int:
    var index = 0
    while a[index] != 0 and b[index] != 0:
        if a[index] < b[index]:
            return -1
        if a[index] > b[index]:
            return 1

        index += 1

    if a[index] != 0 and b[index] == 0:
        return 1

    if a[index] == 0 and b[index] != 0:
        return -1

    return 0

fn str2num(d: Int) -> Int:
    # Convert Hex to deciaml
    if d >= ord("A"):
        return d - ord("A") + 10
    return d - ord("0")

fn print_str(s: PointerString):
    # Print raw byte like <0x0A>
    if (s[1].to_int() == ord("0")) and (s[2].to_int() == ord("x")):
        let d1: Int = s[3].to_int()
        let d2: Int = s[4].to_int()
        print_no_newline(chr(str2num(d1) * 16 + str2num(d2)))
        return
    # Print all chars till null character
    var p: Int = 0
    while s[p].to_int() != 0:
        print_no_newline(chr(s[p].to_int()))
        p += 1

fn wrap(token: PointerString) -> PointerString:
    if string_compare(token, str_to_ptr("\\n")) == 0:
        return str_to_ptr("<0x0A")
    if string_compare(token, str_to_ptr("\\t")) == 0:
        return str_to_ptr("<0x09>")
    if string_compare(token, str_to_ptr("'")) == 0:
        return str_to_ptr("<0x27>")
    elif string_compare(token, str_to_ptr('"')) == 0:
        return str_to_ptr("<0x22>")
    return token