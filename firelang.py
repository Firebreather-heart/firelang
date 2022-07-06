from string_with_arrows import string_with_arrows
import string 

LETTERS = string.ascii_letters
LETTERSDIGITS = LETTERS + string.digits
###############################
#TOKENS
###############################

TT_INT = "INT"
TT_FLOAT = "FLOAT"
TT_PLUS = "PLUS"
TT_MINUS = "MINUS"
TT_MUL = "MUL"
TT_DIV = "DIV"
TT_LPAREN = "LPAREN"
TT_RPAREN = "RPAREN"
TT_EXP = "EXP"
TT_MOD = "MOD"
TT_EOF = "EOF"
TT_ID = "IDENTIFIER"
TT_KEY = 'KEYWORD'
TT_EQ = "EQUALS"
KEYWORDS = [
    "var",
]
#############################
#DIGITS
############################
DIGITS ="0123456789"

class Token:
    def __init__(self, type_, value=None, pos_start= None, pos_end = None) -> None:
        self.type = type_
        self.value = value 

        if pos_start:
            self.pos_start = pos_start.copy()
            self.pos_end =  pos_start.copy()
            self.pos_end.advance()
        if pos_end:
            self.pos_end = pos_end
    def matches(self, type_, value):
        return self.type == type_ and self.value == value
    
    def __repr__(self) -> str:
        if self.value: return f"{self.type}:{self.value}"
        return f'{self.type}'

#########################
#CONTEXT
########################
class Context:
    def __init__(self, display_name: str, parent = None, parent_entry_pos = None) -> None:
        self.display_name = display_name 
        self.parent = parent
        self.parent_entry_pos = parent_entry_pos

########################
#ERRORS
#######################
class Error:
    def __init__(self, pos_start, pos_end,error_name, details) -> None:
        self.pos_start = pos_start
        self.pos_end = pos_end
        self.error_name = error_name
        self.details = details 

    def error_as_string(self)->str:
        result = f'File {self.pos_start.fn}, Line {self.pos_start.ln + 1} \n {self.error_name} :  {self.details}'
        result += '\n\n' + string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)
        return result

class CharError(Error):
    def __init__(self, pos_start, pos_end, details) -> None:
        super().__init__( pos_start, pos_end,"IllegalChar",details)

class ErrorinSyntax(Error):
    def __init__(self, pos_start, pos_end, details='') -> None:
        super().__init__( pos_start, pos_end,"Invalid Syntax",details)

class RTError(Error):
    def __init__(self, pos_start, pos_end, context:Context, details) -> None:
        super().__init__( pos_start, pos_end,"Runtime Error",details)
        self.context = context

    def error_as_string(self) -> str:
        result = self.generate_traceback()
        result += f' {self.error_name} :  {self.details}\n'
        result += '\n\n' + string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)
        return result

    def generate_traceback(self):
        result = ''
        pos = self.pos_start
        ctx:Context = self.context 

        while ctx:
            result = f'File {pos.fn}, line {str(pos.ln +1)} in {ctx.display_name}\n' + result
            pos = ctx.parent_entry_pos
            ctx = ctx.parent
        
        return "ErrorTrace {starting from latest action}:\n" + result


#######################
#RUNTIME RESULT
######################
class RTResults:
    def __init__(self) -> None:
        self.value = None 
        self.error = None 

    def register(self, res):
        if res.error: self.error = res.error 
        return res.value 
    
    def success(self, value):
        self.value = value 
        return self 
    def failure(self, error):
        self.error = error 
        return self


################
#POSITION
################
class Position:
    def __init__(self, idx: int, ln: int, col:int, fn:str, ftxt: str) -> None:
        self.fn = fn 
        self.ftxt = ftxt
        self.idx = idx 
        self. ln = ln 
        self.col = col 

    def advance(self, current_char=None):
        self.idx += 1
        self.col += 1

        if current_char == "\n":
            self.idx += 1 
            self.col = 0
        
        return self 
    
    def copy(self):
        return Position(self.idx , self.ln, self.col, self.fn, self.ftxt)

###############################
#LEXER
###############################

class Lexer:
    def __init__(self,fn:str,text) -> None:
        self.fn = fn
        self.text = text 
        self.pos = Position(-1, 0, -1, fn, text)
        self.current_char = None 
        self.advance()
    
    def advance(self):
        self.pos.advance(self.current_char)
        self.current_char = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None 

    def make_tokens(self):
        tokens = []
        
        while self.current_char != None:
            if self.current_char in ' \t':
                self.advance()
            
            elif self.current_char in DIGITS:
                tokens.append(self.make_number())

            elif self.current_char == '+':
                tokens.append(Token(TT_PLUS, pos_start=self.pos))
                self.advance()
            elif self.current_char == '-':
                tokens.append(Token(TT_MINUS, pos_start=self.pos))
                self.advance()


            elif self.current_char == '/':
                tokens.append(Token(TT_DIV, pos_start=self.pos))
                self.advance()


            elif self.current_char == '*':
                tokens.append(Token(TT_MUL, pos_start=self.pos))
                self.advance()


            elif self.current_char == '(':
                tokens.append(Token(TT_LPAREN, pos_start=self.pos))
                self.advance()

            elif self.current_char == ')':
                tokens.append(Token(TT_RPAREN, pos_start=self.pos))
                self.advance()   
            
            elif self.current_char == '^':
                tokens.append(Token(TT_EXP, pos_start=self.pos))
                self.advance()   
            
            elif self.current_char == '=':
                tokens.append(Token(TT_EQ, pos_start=self.pos))
                self.advance() 

            elif self.current_char in LETTERS:
                tokens.append(self.make_identifier())
                self.advance() 
            
            elif self.current_char == '%':
                tokens.append(Token(TT_MOD, pos_start=self.pos))
                self.advance()   
            
            else:
                pos_start = self.pos.copy()
                char = self.current_char
                self.advance()
                return [], CharError(pos_start, self.pos," ' " + char + " ' ")
        tokens.append(Token(TT_EOF, pos_start=self.pos))
        return tokens,None

    def make_number(self):
        num_str = ''
        dot_count = 0
        pos_start = self.pos.copy()

        while self.current_char != None and self.current_char in DIGITS + ".":
            if self.current_char == '.':
                if dot_count == 1: break 
                dot_count += 1 
                num_str += '.'
            else: num_str += self.current_char
            self.advance()

        if dot_count == 0:
            return Token(TT_INT, int(num_str), pos_start,self.pos)
        else:
            return Token(TT_FLOAT, float(num_str), pos_start,self.pos)
        
    def make_identifier(self,):
        id_str=''
        pos_start = self.pos.copy()

        while self.current_char != None and self.current_char in LETTERSDIGITS + '_':
            id_str += self.current_char
            self.advance()
        tok_type = TT_KEY if id_str in KEYWORDS else TT_ID
        return Token(tok_type, id_str, pos_start, self.pos)


###############
#NODE
###############

class NumberNode:
    def __init__(self, tok) -> None:
        self.tok = tok 
        self.pos_start = self.tok.pos_start
        self.pos_end = self.tok.pos_end
    
    def __repr__(self) -> str:
        return f'{self.tok}'

class BinOpNode:
    def __init__(self, leftnode, op_tok, rightnode) -> None:
        self.leftnode = leftnode 
        self.op_tok = op_tok 
        self.rightnode = rightnode

        self.pos_start = self.leftnode.pos_start
        self.pos_end = self.rightnode.pos_end
    
    def __repr__(self) -> str:
        return f"({self.leftnode}, {self.op_tok}, {self.rightnode})"

class UnaryNode:
    def __init__(self, op_tok, node) -> None:
        self.op_tok = op_tok 
        self.node = node

        self.pos_start = self.op_tok.pos_start
        self.pos_end = self.node.pos_end
    
    def __repr__(self) -> str:
        return f"( {self.op_tok}, {self.node})"


#########################################################
#PARSE RESULTS
########################################################

class ParseResults:
    def __init__(self) -> None:
        self.error = None 
        self.node = None

    def register(self, res):
        if isinstance(res, ParseResults):
            if res.error: self.error = res.error 
            return res.node 
        return res
    
    def success(self, node):
        self.node = node 
        return self 

    def failure(self, error):
        self.error = error 
        return self

######################
#PARSER
#####################

class Parser:
    def __init__(self, tokens) -> None:
        self.tokens = tokens 
        self.tok_idx = -1
        self.advance()
    
    def advance(self):
        self.tok_idx += 1
        if self.tok_idx < len(self.tokens):
            self.current_tok = self.tokens[self.tok_idx]
        return self.current_tok
    
    def binOp(self,func, ops):
        res = ParseResults()
        left = res.register(func())

        if res.error: return res 
        
        while self.current_tok.type in ops:
            op_tok = self.current_tok
            self.advance()
            right = res.register(func())
            if res.error: return res 
            left = BinOpNode(left, op_tok, right)
        return res.success(left)
    
    ####################################
    def parse(self):
        res = self.expr()
        if not res.error and self.current_tok.type != TT_EOF:
            return res.failure(
                ErrorinSyntax(
                    self.current_tok.pos_start, self.current_tok.pos_end, "Invalid Operator"
                )
            )
        return res

    def factor(self):
        res = ParseResults()
        tok = self.current_tok

        if tok.type in (TT_PLUS, TT_MINUS,TT_MOD,TT_EXP):
            res.register(self.advance())
            factor = res.register(self.factor())

            if res.error: return res
            return res.success(UnaryNode(tok, factor))

        elif tok.type in (TT_INT, TT_FLOAT):
            res.register(self.advance())
            return res.success(NumberNode(tok))
        
        elif tok.type == TT_LPAREN:
            res.register(self.advance())
            expr = res.register(self.expr())

            if res.error : return res 
            if self.current_tok.type == TT_RPAREN:
                res.register(self.advance())
                return res.success(expr)
            else:
                return res.failure(ErrorinSyntax(
                    tok.pos_start, tok.pos_end, f"Expected ')'  not {tok.type}"
                ))
            

            

        return res.failure(ErrorinSyntax(
            tok.pos_start, tok.pos_end, f"Expected int or float not {tok.type}"
        ))
    
    
    def term(self):
        return self.binOp(self.factor, (TT_MUL, TT_DIV))
    
    
    def expr(self):
        res = ParseResults()

        if self.current_tok.matches(TT_KEY, 'var'):
            res.register(self.advance())
        return self.binOp(self.term, (TT_PLUS,TT_MINUS, TT_MOD, TT_EXP))
##############################3
#NUMBER
##############################

class Number:
    def __init__(self, value) -> None:
        self.value = value 
        self.set_pos()
        self.set_context()
    
    def set_pos(self, pos_start = None, pos_end = None):
        self.pos_start = pos_start 
        self.pos_end = pos_end
        return self 

    def set_context(self, context = None):
        self.context = context
        return self
    
    def added_to(self, other):
        if isinstance(other, Number):
            return Number(self.value + other.value).set_context(self.context), None

    def sub_from(self, other):
        if isinstance(other, Number):
            return Number(self.value - other.value).set_context(self.context), None
    
    def div_by(self, other):
        if isinstance(other, Number):
            if other.value == 0:
                return None, RTError(
                    other.pos_start, other.pos_end, self.context, "Indeterminate Result: Division by Zero", 
                )
            return Number(self.value / other.value).set_context(self.context), None
    
    def mul_by(self, other):
        if isinstance(other, Number):
            return Number(self.value * other.value).set_context(self.context), None

    def exp(self, other):
        if isinstance(other, Number):
            return Number(self.value ** other.value).set_context(self.context), None

    def mod(self, other):
        if isinstance(other, Number):
            return Number(self.value % other.value).set_context(self.context), None
    
    def __repr__(self) -> str:
        return str(self.value)



        

#####################################
#INTERPRETER
#####################################
class Interpreter:
    def visit(self, node, context:Context):
        method_name = f'burn_{type(node).__name__}'
        method = getattr(self, method_name, self.no_burn_method)
        return method(node, context)

    def no_burn_method(self, node, context):
        raise Exception(f"No burn_{type(node).__name__} method defined")

    def burn_NumberNode(self, node,  context):
        return RTResults().success(
         Number(node.tok.value).set_context(context).set_pos(node.pos_start, node.pos_end)
        )

    def burn_BinOpNode(self, node, context):
        res = RTResults()
        left = res.register(self.visit(node.leftnode, context))
        if res.error: return res
        right = res.register(self.visit(node.rightnode, context))
        if res.error: return res
        result = None
        if node.op_tok.type == TT_PLUS:
            result, error = left.added_to(right)
        elif node.op_tok.type == TT_MINUS:
            result, error = left.sub_from(right)
        elif node.op_tok.type == TT_DIV:
            result, error = left.div_by(right)
        elif node.op_tok.type == TT_EXP:
            result, error = left.exp(right)
        elif node.op_tok.type == TT_MUL:
            result, error = left.mul_by(right)
        elif node.op_tok.type == TT_MOD:
            result, error = left.mod(right) 
        if error:
            return res.failure(error)
        else:
            return res.success(result.set_pos(node.pos_start, node.pos_end)
)
    def burn_UnaryNode(self, node, context):
        res = RTResults()
        number = res.register(self.visit(node.node, context))
        if res.error: return res 

        error = None 
        if node.op_tok.type == TT_MINUS:
            number, error = number.mul_by(Number(-1))
        if error: return res.failure(error)

        else:
            return res.success(number.set_pos(node.pos_start, node.pos_end))
        

###################
#RUN
#################
def run(fn,text):
    lexer = Lexer(fn,text)
    tokens , errors = lexer.make_tokens()

    if errors: return None,errors 

    #Generate ASI
    parser = Parser(tokens)
    ast = parser.parse()
    if ast.error: return None, ast.error

    context = Context('<program>')

    interpreter = Interpreter()
    result = interpreter.visit(ast.node,context )


    return result.value, result.error