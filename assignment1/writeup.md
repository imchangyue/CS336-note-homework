### 1
![alt text](/image/image.png)
 - (a)ord(chr(0)) = 0
 - (b)字符的 __repr__() 表示是: ''\x00'';字符的 print() 打印表示是: '' 
 - (c)在普通的string中是`__repr()__`表示，但是在print()中是空字符表示


### 2
![alt text](/image/image-1.png)
 - 对于同样的语句"hello! こんにちは!",用utf-8,utf-16,utf-32分别解码得到的len(list(encoded_string))分别是23，28，56，用utf-8解码出的长度最小
 - `"".join([bytes([b]).decode("utf-8") for b in bytestring])`试图逐个字节地进行解码。这是导致错误的关键原因。UTF-8 是一种可变长度的编码。这意味着一个 Unicode 字符可能由一个、两个、三个甚至四个字节组成。例如：英文字符（如 'h', 'e', 'l', 'l', 'o', '!'）通常只占一个字节。日文、中文、韩文等字符通常会占两个或三个字节（甚至更多）。例如，日文的 こ (ko) 在 UTF-8 中通常是 3 个字节（e3 81 93）。
 - **0xC0 0x80**:  0xC0 和 0xC1 是无效的起始字节，因为它们本应表示 2 字节序列的开始，但它们所指示的码点范围是空的。紧随其后的任何字节（例如 0x80）都无法完成一个有效的 UTF-8 字符序列，因为 0x80 到 0xBF 范围的字节是 UTF-8 多字节序列中的后续字节，不能作为序列的第一个字节。因此，0xC0 0x80 这个组合是无效的 UTF-8 序列，无法解码成任何 Unicode 字符。