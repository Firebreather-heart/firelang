def string_with_arrows(text:str, pos_start: int, pos_end: int) -> str:
    result = ''

    idx_start = max(text.rfind('\n',0, pos_start.idx), 0)
    idx_end = text.find('\n', idx_start + 1)
    if idx_end < 0: idx_end = len(text)

    line_count = pos_end.ln - pos_start.ln + 1
    for i in range(line_count):
         #calc line cols
        line = text[idx_start:idx_end]
        col_start = pos_start.col if i == 0 else 0 
        col_end = pos_end.col if i == line_count - 1 else len(line) - 1

        #append to result 
        result += line + '\n'
        result += ' ' * col_start + '^' * (col_end - col_start)

        #recalculate indices
        idx_start = idx_end 
        idx_end = text.find('\n', idx_start + 1)
        if idx_end < 0: idx_end = len(text)
        
    return result.replace('\t', '')