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
TT_EE = 'EE'
TT_LE = 'LE'
TT_GE ='GE'
TT_NE ='NE'
TT_GT = 'GT'
TT_LT = 'LT'
TT_LB = "LBRACKET"
TT_RB = "RBRACKET"
TT_CM = 'COMMA'
TT_AR = 'ARROW'
TT_SC = 'SCOLON'

KEYWORDS = [
    "var",
    "and",
    "or",
    "not",
    'if',
    #'is',
    'elif',
    'else',
    'for',
    'while',
    'step',
    'fxn',
]
TYPES =[
    "int",
    'float',
    'string',
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
        self.symbol_table = None

########################
#ERRORS
#######################
class Error:
    """Base Error class"""
    def __init__(self, pos_start, pos_end,error_name, details) -> None:
        self.pos_start = pos_start
        self.pos_end = pos_end
        self.error_name = error_name
        self.details = details 

    def error_as_string(self)->str:
        result = f'burnt\n File {self.pos_start.fn}, Line {self.pos_start.ln + 1} \n {self.error_name} :  {self.details}'
        result += '\n\n' + string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)
        return result

class CharError(Error):
    def __init__(self, pos_start, pos_end, details) -> None:
        super().__init__( pos_start, pos_end,"IllegalChar",details)

class ExpectedCharError(Error):
    def __init__(self, pos_start, pos_end, details) -> None:
        super().__init__( pos_start, pos_end,"Expected Character",details)

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
        
        return "Burnt!\n BurnTrace {starting from latest action}:\n" + result


#######################
#RUNTIME RESULT
######################
class RTResults:
    """A class for handling the runtime results.\nIt's basically used as a wrapper for other functions.  """
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



###############################
#LEXER
###############################

class Lexer:
    """The lexicon class  for defining the acceptable commands and characters"""
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
                tokens.append(self.make_minus_or_tokens())

            elif self.current_char == '{':
                tokens.append(Token(TT_LB, value='{', pos_start=self.pos))
                self.advance()

            elif self.current_char == '}':
                tokens.append(Token(TT_RB,value='}', pos_start=self.pos))
                self.advance()
            
            elif self.current_char == ',':
                tokens.append(Token(TT_CM,value=',', pos_start=self.pos))
                self.advance()

            elif self.current_char == " ":
                self.advance()

            elif self.current_char == '/':
                tokens.append(Token(TT_DIV, pos_start=self.pos))
                self.advance()
            
            elif self.current_char == ';':
                tokens.append(Token(TT_SC,value=';', pos_start=self.pos))
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
                tokens.append(self.make_equals())
                self.advance() 

            elif self.current_char == '<':
                tokens.append(self.make_less_than())
                self.advance() 
            elif self.current_char == '>':
                tokens.append(self.make_greater_than())
                self.advance() 
            elif self.current_char in LETTERS:
                tokens.append(self.make_identifier())
                self.advance() 
            
            elif self.current_char == '%':
                tokens.append(Token(TT_MOD, pos_start=self.pos))
                self.advance()   
            
            elif self.current_char == '!':
                tok, error = self.make_not_equals()
                if error : return [], error 
                tokens.append(tok)
            
            else:
                pos_start = self.pos.copy()
                char = self.current_char
                self.advance()
                return [], CharError(pos_start, self.pos," ' " + char + " ' ")

        tokens.append(Token(TT_EOF, pos_start=self.pos))
        return tokens,None

    def make_minus_or_tokens(self):
        tok_type = TT_MINUS
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == '>':
            self.advance()
            tok_type = TT_AR
        return Token(tok_type, pos_start= pos_start, pos_end =self.pos)
    def make_not_equals(self):
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == '=':
            self.advance()
            return Token(TT_NE, pos_start, self.pos), None
        self.advance()
        return None, ExpectedCharError(pos_start, self.pos, "'=' after '!'")

    def make_equals(self):
        tok_type = TT_EQ 
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == "=":
            self.advance()
            tok_type = TT_EE 
        return Token(tok_type, pos_start, self.pos)
    
    def make_greater_than(self):
        tok_type = TT_GT
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == "=":
            self.advance()
            tok_type = TT_GE 
        return Token(tok_type, pos_start, self.pos)
    
    def make_less_than(self):
        tok_type = TT_LT 
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == "=":
            self.advance()
            tok_type = TT_LE
        return Token(tok_type, pos_start, self.pos)

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

class MasterType:
    def __init__(self) -> None:
        self.set_pos()
        self.set_context()
    
    def set_pos(self, pos_start=None, pos_end = None):
        self.pos_start = pos_start 
        self.pos_end = pos_end
        return self 
    
    def set_context(self,context = None):
        self.context = context 
        return self 
    def added_to(self, other):
        return None, self.illegal_operation(other)
    def subbbed_by(self, other):
        return None, self.illegal_operation(other)
    def mul_by(self, other):
        return None, self.illegal_operation(other)
    def div_by(self, other):
        return None, self.illegal_operation(other)
    def mod_by(self, other):
        return None, self.illegal_operation(other)
    def pow_by(self, other):
        return None, self.illegal_operation(other)
    def get_comparison_eq(self, other):
        return None, self.illegal_operation(other)
    def get_comparison_ne(self, other):
        return None, self.illegal_operation(other)
    def get_comparison_lt(self, other):
        return None, self.illegal_operation(other)
    def get_comparison_gt(self, other):
        return None, self.illegal_operation(other)
    def get_comparison_le(self, other):
        return None, self.illegal_operation(other)
    def get_comparison_ge(self, other):
        return None, self.illegal_operation(other)
    def anded_by(self, other):
        return None, self.illegal_operation(other)
    def ored_by(self, other):
        return None, self.illegal_operation(other)
    def notted(self, other):
        return None, self.illegal_operation(other)
    def execute(self, other):
        return None, self.illegal_operation(other)
    def copy(self):
        raise Exception("No copy method defined")
    def is_true(self):
        return False 
    def illegal_operation(self, other=None):
        if not other: other = self 
        return RTError(
            self.pos_start, self.pos_end, self.context, "Illegal Operation"
        )
        
###############
#NODE
###############
class Node:
    """A generic class for future tokens, args include tok, pos_start, pos_end, var_name_tok,op_tok,\n
    left_node, right_node, cases, else_cases,node"""
    def __init__(self, **kwargs) -> None:
        self.kws = kwargs
        self.tok :  Token = kwargs.get('tok')
        self.pos_start = kwargs.get('pos_start')
        self.pos_end = kwargs.get('pos_end')
        self.var_name_tok: Token = kwargs.get('var_name_token')
        self.op_tok: Token = kwargs.get('op_tok')
        self.left_node = kwargs.get('left_node')
        self.right_node = kwargs.get('right_node')
        self.cases = kwargs.get('cases')
        self.else_cases = kwargs.get('else_cases')
        self.node = kwargs.get('node')
        self.cleanup()

    def cleanup(self):
        for key in self.kws.keys():
            try:
                attr = getattr(self, key)
                if attr is None:
                    delattr(self, key)
            except Exception:
                pass
        return self

class FuncDefNode:
    def __init__(self, var_name_tok:Token, arg_name_toks, body_node: Node):
        self.var_name_tok = var_name_tok
        self.arg_name_toks = arg_name_toks 
        self.body_node = body_node 

        if self.var_name_tok:
            self.pos_start = self.var_name_tok.pos_start
        elif len(self.arg_name_toks) > 0:
            self.pos_start = self.arg_name_toks[0].pos_start
        else:
            self.pos_start = self.body_node.pos_start
        self.pos_end = self.body_node.pos_end
class CallNode:
    def __init__(self, node_to_call:Node, arg_nodes) -> None:
        self.node_to_call = node_to_call
        self.arg_nodes = arg_nodes

        self.pos_start = self.node_to_call.pos_start
        if len(self.arg_nodes) > 0:
            self.pos_end = self.arg_nodes[len(self.arg_nodes) -1].pos_end
        else:
            self.pos_end = self.node_to_call.pos_end

          

class ForNode(Node):
    def __init__(self, **kwargs) -> None:
        super(ForNode,self).__init__(**kwargs)
        self.start_value_node = kwargs.get('start_value_node')
        self.end_value_node = kwargs.get('end_value_node')
        self.step_value_node = kwargs.get('step_value_node')
        self.body_node = kwargs.get('body_node')

        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.body_node.pos_end

class WhileNode(Node):
    def __init__(self, **kwargs) -> None:
        super(WhileNode, self).__init__(**kwargs)
        self.condition_node = kwargs.get('condition_node')
        self.body_node = kwargs.get('body_node')

        self.pos_start = self.condition_node.pos_start
        self.pos_end = self.body_node.pos_end



class NumberNode:
    def __init__(self, tok:Token) -> None:
        self.tok = tok 
        self.pos_start = self.tok.pos_start
        self.pos_end = self.tok.pos_end
    
    def __repr__(self) -> str:
        return f'{self.tok}'

class VarAccessNode:
    def __init__(self, var_name_tok: Token) -> None:
        self.var_name_tok = var_name_tok

        self.pos_start = self.var_name_tok.pos_start
        self.pos_end  = self.var_name_tok.pos_end

class VarAssignedNode:
    def __init__(self, var_name_tok:Token, value_node) -> None:
        self.var_name_tok = var_name_tok
        self.value_node = value_node

        self.pos_start = self.var_name_tok.pos_start
        self.pos_end  = self.value_node.pos_end
        
class BinOpNode:
    def __init__(self, leftnode, op_tok, rightnode) -> None:
        self.leftnode = leftnode 
        self.op_tok = op_tok 
        self.rightnode = rightnode

        self.pos_start = self.leftnode.pos_start
        self.pos_end = self.rightnode.pos_end
    
    def __repr__(self) -> str:
        return f"({self.leftnode}, {self.op_tok}, {self.rightnode})"

class IFNode:
    def __init__(self, cases, else_case) -> None:
        self.cases = cases 
        self.else_case = else_case

        self.pos_start = self.cases[0][0].pos_start
        self.pos_end = (self.else_case or self.cases[len(self.cases) - 1][0]).pos_end


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
        self.advance_count = 0

    def register(self, res):
            self.advance_count += res.advance_count
            if res.error: self.error = res.error 
            return res.node 

    def register_advancement(self):
        self.advance_count += 1
        pass
    
    def success(self, node):
        self.node = node 
        return self 

    def failure(self, error):
        if not self.error or self.advance_count == 0:
            self.error = error 
        return self

######################
#PARSER
#####################

class Parser:
    def __init__(self, tokens:Token) -> None:
        self.tokens = tokens 
        self.tok_idx = -1
        self.advance()
    
    def advance(self):
        self.tok_idx += 1
        if self.tok_idx < len(self.tokens):
            self.current_tok = self.tokens[self.tok_idx]
        return self.current_tok
    
    def binOp(self,func, ops, funcb=None):
        
        if funcb is None:
            funcb = func

        res = ParseResults()
        left = res.register(func())

        if res.error: return res 
        
        while self.current_tok.type in ops or (self.current_tok.type, self.current_tok.value) in ops:
            op_tok = self.current_tok
            self.advance()
            right = res.register(funcb())
            if res.error: return res 
            left = BinOpNode(left, op_tok, right)
        return res.success(left)
    
    def if_expr(self):
        res = ParseResults()
        cases = []
        else_case = None 

        if not self.current_tok.matches(TT_KEY, 'if'):
            return res.failure(
                ErrorinSyntax(
                    self.current_tok.pos_start, self.current_tok.pos_end, "Expected 'if'"
                )
            )
        res.register_advancement()
        self.advance()

        condition = res.register(self.expr()) 

        if res.error: return res 

        if not self.current_tok.matches(TT_LB, '{'):
            return res.failure(
                ErrorinSyntax(
                    self.current_tok.pos_start, self.current_tok.pos_end, "Expected '{' " + f"instead got {self.current_tok.value}"
                )
            )
        res.register_advancement()
        self.advance()

        expr = res.register(self.expr())
        if res.error: return res 
        cases.append((condition, expr))

        if not self.current_tok.matches(TT_RB, '}'):
            return res.failure(
                ErrorinSyntax(
                    self.current_tok.pos_start, self.current_tok.pos_end, "Expected '}'"
                )
            )
        res.register_advancement()
        self.advance()
        

        while self.current_tok.matches(TT_KEY, 'elif'):
            res.register_advancement()
            self.advance()
        
            condition = res.register(self.expr())
            if res.error: return res 

            if not self.current_tok.matches(TT_LB, '{'):
                return res.failure(
                    ErrorinSyntax(
                        self.current_tok.pos_start, self.current_tok.pos_end, "Expected '{'"
                    )
                )
            res.register_advancement()
            self.advance()

            expr = res.register(self.expr())
            if res.error: return res 
            cases.append((condition, expr))

            if not self.current_tok.matches(TT_RB, '}'):
                return res.failure(
                ErrorinSyntax(
                    self.current_tok.pos_start, self.current_tok.pos_end, "Expected '}'"
                )
            )
            res.register_advancement()
            self.advance()

            ###########
        if  self.current_tok.matches(TT_KEY, 'else'):
            res.register_advancement()
            self.advance()
            if not self.current_tok.matches(TT_LB, '{'):
                return res.failure(
                ErrorinSyntax(
                    self.current_tok.pos_start, self.current_tok.pos_end, "Expected '{'"
                )
            )
            res.register_advancement()
            self.advance()

            else_case = res.register(self.expr())
            if res.error: return res

            if not self.current_tok.matches(TT_RB, '}'):
                return res.failure(
                ErrorinSyntax(
                    self.current_tok.pos_start, self.current_tok.pos_end, "Expected '}'"
                )
            )
            res.register_advancement()
            self.advance()
        return res.success(IFNode(cases, else_case))
    
    def call(self):
        res = ParseResults()
        atom = res.register(self.atom())
        if res.error: return res

        if self.current_tok.type == TT_LPAREN:
            res.register_advancement()
            self.advance()
            arg_nodes = []

            if self.current_tok.type == TT_RPAREN:
                res.register_advancement()
                self.advance()
            else:
                arg_nodes.append(res.register(self.expr()))     
                if res.error: 
                    return res.failure(
                        ErrorinSyntax(
                            self.current_tok.pos_start, self.current_tok.pos_end, f"Expected 'var' int, float, variable, 'operator' or '(' not {self.current_tok.value}"
                                )
                            ) 
                while self.current_tok.type ==  TT_CM:
                    res.register_advancement()
                    self.advance()

                    arg_nodes.append(res.register(self.expr()))
                    if res.error: return res 
                if self.current_tok.type != TT_RPAREN:
                    return res.failure(ErrorinSyntax(
                        self.current_tok.pos_start, self.current_tok.pos_end, f"Expected ',' or ')' instead got {self.current_tok.value}"
                    ))
                res.register_advancement()
                self.advance()
            return res.success(CallNode(atom,arg_nodes))
        return res.success(atom)
                


    
    ####################################
    def atom(self):
        res = ParseResults()
        tok = self.current_tok

        if tok.type in (TT_INT, TT_FLOAT,):
            res.register_advancement()
            self.advance()
            return res.success(NumberNode(tok))

        elif tok.type  == TT_ID:
            res.register_advancement()
            self.advance()
            return res.success(VarAccessNode(tok))
            
        elif tok.type == TT_LPAREN:
            res.register_advancement()
            self.advance()
            expr = res.register(self.expr())

            if res.error : return res 
            if self.current_tok.type == TT_RPAREN:
                res.register_advancement()
                self.advance()
                return res.success(expr)
            else:
                return res.failure(ErrorinSyntax(
                    tok.pos_start, tok.pos_end, f"Expected ')'  not {tok.type}"
                ))
        elif tok.matches(TT_KEY, 'if'):
            if_expr = res.register(self.if_expr())
            if res.error:return res 
            return res.success(if_expr)

        elif tok.matches(TT_KEY, 'for'):
            for_expr = res.register(self.for_expr())
            if res.error:return res 
            return res.success(for_expr)
            
        elif tok.matches(TT_KEY, 'while'):
            while_expr = res.register(self.while_expr())
            if res.error:return res 
            return res.success(while_expr)
        
        elif tok.matches(TT_KEY, 'fxn'):
            func_def = res.register(self.func_def())
            if res.error:return res 
            return res.success(func_def)

        return res.failure(ErrorinSyntax(
            tok.pos_start, tok.pos_end, f"Expected int, float, variable, 'operator' or '(', 'for', 'while', 'if', 'fxn' not {tok.type}"
        ))

        

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

        if tok.type in (TT_PLUS, TT_MINUS, ):
            res.register_advancement()
            self.advance()
            factor = res.register(self.factor())

            if res.error: return res
            return res.success(UnaryNode(tok, factor))

        return self.power()
    
    def power(self):
        return self.binOp(self.atom,(TT_EXP, TT_MOD),self.factor)
    
    
    def term(self):
        return self.binOp(self.factor, (TT_MUL, TT_DIV))
    
    def comp_expr(self):
        res = ParseResults()
        if self.current_tok.matches(TT_KEY, 'NOT'):
            op_tok = self.current_tok
            res.register_advancement()
            self.advance()

            node = res.register(self.comp_expr())
            if res.error: return res 
            return res.success(UnaryNode(op_tok, node))
        node = res.register(self.binOp(self.arith_expr, (TT_EE,TT_GT, TT_LT,TT_LE, TT_GE, TT_NE,)))
        if res.error:return res.failure(
            ErrorinSyntax(
                self.current_tok.pos_start, self.current_tok.pos_end, f"Expected int, float, variable, 'operator', '(', 'not', not {self.current_tok.value}"
            )
        )
        return res.success(node)
    def arith_expr(self):
        return self.binOp(self.term, (TT_PLUS, TT_MINUS, ))
    def func_def(self):
        res = ParseResults()
        if not self.current_tok.matches(TT_KEY, 'fxn'):
            return res.failure(
                ErrorinSyntax(
                    self.current_tok.pos_start, self.current_tok.pos_end, "Expected 'fxn' " + f"instead got {self.current_tok.value}"
                )
            )
        res.register_advancement()
        self.advance()
        if self.current_tok.type == TT_ID:
            var_name_tok = self.current_tok
            res.register_advancement()
            self.advance()
            if self.current_tok.type != TT_LPAREN:
                return res.failure(ErrorinSyntax(
                self.current_tok.pos_start, self.current_tok.pos_end, f"Expected ( instead got {self.current_tok.value}"
            ))
        else:
            var_name_tok = None 
            if self.current_tok.type != TT_LPAREN:
                return res.failure(ErrorinSyntax(
                self.current_tok.pos_start, self.current_tok.pos_end, f"Expected ( instead got {self.current_tok.value}"
            ))
        res.register_advancement()
        self.advance()
        arg_name_toks = []
        
        if self.current_tok.type == TT_ID:
            arg_name_toks.append(self.current_tok)
            res.register_advancement()
            self.advance()

            while self.current_tok.type == TT_CM:
                res.register_advancement()
                self.advance()

                if self.current_tok.type != TT_ID:
                    return res.failure(
                        ErrorinSyntax(
                    self.current_tok.pos_start, self.current_tok.pos_end, "Expected 'identifier' "
                    )
                )
                arg_name_toks.append(self.current_tok)
                res.register_advancement()
                self.advance()
            if self.current_tok.type != TT_RPAREN:
                return res.failure(ErrorinSyntax(
                self.current_tok.pos_start, self.current_tok.pos_end, f"Expected ',' or ')' instead got {self.current_tok.value}"
            ))
        else:
            if self.current_tok.type != TT_RPAREN:
                return res.failure(ErrorinSyntax(
                self.current_tok.pos_start, self.current_tok.pos_end, f"Expected identifier or ')' instead got {self.current_tok.value}"
            ))
        res.register_advancement()
        self.advance()

        if self.current_tok.type != TT_AR:
                return res.failure(ErrorinSyntax(
                self.current_tok.pos_start, self.current_tok.pos_end, f"Expected '->' instead got {self.current_tok.value}"
            ))
        res.register_advancement()
        self.advance()
        node_to_return = res.register(self.expr())
        if res.error: return res
        return res.success(
            FuncDefNode(var_name_tok, arg_name_toks,node_to_return)
        )
            




    def for_expr(self):
        res = ParseResults()

        if not self.current_tok.matches(TT_KEY, 'for'):
            return res.failure(
                ErrorinSyntax(
                    self.current_tok.pos_start, self.current_tok.pos_end, "Expected 'for' " + f"instead got {self.current_tok.value}"
                )
            )
        res.register_advancement()
        self.advance()

        if self.current_tok.type != TT_ID:
            return res.failure(ErrorinSyntax(
                self.current_tok.pos_start, self.current_tok.pos_end, f"Expected variable name instead got {self.current_tok.value}"
            ))
        var_name = self.current_tok
        res.register_advancement()
        self.advance()

        if self.current_tok.type != TT_EQ:
            return res.failure(ErrorinSyntax(
                self.current_tok.pos_start, self.current_tok.pos_end, f"Expected '=' instead got {self.current_tok.value}"
            ))
        
        res.register_advancement()
        self.advance()

        start_value = res.register(self.expr())
        if res.error: return res 

        if self.current_tok.type != TT_SC:
            return res.failure(ErrorinSyntax(
                self.current_tok.pos_start, self.current_tok.pos_end, f"Expected ';' instead got {self.current_tok.value}"
            ))
        res.register_advancement()
        self.advance()

        if self.current_tok.type == TT_ID:
           # return res.failure(ErrorinSyntax(
            #    self.current_tok.pos_start, self.current_tok.pos_end, f"Expected variable name instead got {self.current_tok.value}"
            #))
            res.register_advancement()
            self.advance()

            if self.current_tok.type != TT_EQ:
                return res.failure(ErrorinSyntax(
                self.current_tok.pos_start, self.current_tok.pos_end, f"Expected '=' instead got {self.current_tok.value}"
                ))
        
            res.register_advancement()
            self.advance()

        end_value = res.register(self.expr())
        if res.error: return res 

        if self.current_tok.type == TT_SC:
            #return res.failure(ErrorinSyntax(
            #    self.current_tok.pos_start, self.current_tok.pos_end, f"Expected ';' instead got {self.current_tok.value}"
            #))
            res.register_advancement()
            self.advance()

            if  self.current_tok.matches(TT_KEY, 'step'):
            
                res.register_advancement()
                self.advance()

            

                step_value = res.register(self.expr())
                if res.error:return res 
        else:
            step_value = None 
        
        if not self.current_tok.matches(TT_LB, '{'):
                return res.failure(
                ErrorinSyntax(
                    self.current_tok.pos_start, self.current_tok.pos_end, "Expected '{'"
                )
            )
        res.register_advancement()
        self.advance()

        body = res.register(self.expr())
        if res.error: return res

        if not self.current_tok.matches(TT_RB, '}'):
            return res.failure(
                ErrorinSyntax(
                    self.current_tok.pos_start, self.current_tok.pos_end, "Expected '}'"
                )
            )
        res.register_advancement()
        self.advance()
        return res.success(ForNode(var_name_token = var_name, start_value_node = start_value, end_value_node = end_value, body_node=body,step_value_node=step_value))

    def while_expr(self):
        res = ParseResults()

        if not self.current_tok.matches(TT_KEY, 'while'):
            return res.failure(
                ErrorinSyntax(
                    self.current_tok.pos_start, self.current_tok.pos_end, f"Expected 'while'"
                )
            )
        res.register_advancement()
        self.advance()

        condition = res.register(self.expr())
        if res.error: return res

        if not self.current_tok.matches(TT_LB, '{'):
                return res.failure(
                ErrorinSyntax(
                    self.current_tok.pos_start, self.current_tok.pos_end, "Expected '{'"
                )
            )
        res.register_advancement()
        self.advance()

        body = res.register(self.expr())
        if res.error: return res

        if not self.current_tok.matches(TT_RB, '}'):
            return res.failure(
                ErrorinSyntax(
                    self.current_tok.pos_start, self.current_tok.pos_end, "Expected '}'"
                )
            )
        res.register_advancement()
        self.advance()
        return res.success(WhileNode(condition_node = condition, body_node = body))






    def expr(self):
        res = ParseResults()

        if self.current_tok.matches(TT_KEY, 'var'):
            res.register_advancement()
            self.advance()

            if self.current_tok.type != TT_ID:
                return res.failure(ErrorinSyntax(
                    self.current_tok.pos_start, self.current_tok.pos_end, f"Expected variable name, instead got {self.current_tok.value}"
                ))
            var_name = self.current_tok
            res.register_advancement()
            self.advance()

            if self.current_tok.type != TT_EQ:
                return res.failure(ErrorinSyntax(
                    self.current_tok.pos_start, self.current_tok.pos_end, f"Expected assignment operator '=', instead got {self.current_tok.value}"
                ))
            res.register_advancement()
            self.advance()
            expr = res.register(self.expr())
            if res.error: return res 
            return res.success(VarAssignedNode(var_name, expr))
        node = res.register(self.binOp(self.comp_expr, ((TT_KEY,"and"), (TT_KEY, "or"))))
        if res.error: return res.failure(
            ErrorinSyntax(
                self.current_tok.pos_start, self.current_tok.pos_end, f"Expected 'var' int, float, variable, 'operator', 'if', 'for', 'while', 'fxn', or '(' not {self.current_tok.value}"
            )
        ) 

        return res.success(node)

##############################3
#NUMBER
##############################

class Number(MasterType):
    def __init__(self, value) -> None:
        super().__init__()
        self.value = value 
    
    def added_to(self, other):
        if isinstance(other, Number):
            return Number(self.value + other.value).set_context(self.context), None
        else:
            return None, MasterType.illegal_operation(self.pos_start, self.pos_end)

    def sub_from(self, other):
        if isinstance(other, Number):
            return Number(self.value - other.value).set_context(self.context), None
        else:
            return None, MasterType.illegal_operation(self.pos_start, self.pos_end)
    
    def div_by(self, other):
        if isinstance(other, Number):
            if other.value == 0:
                return None, RTError(
                    other.pos_start, other.pos_end, self.context, "Indeterminate Result: Division by Zero", 
                )
            return Number(self.value / other.value).set_context(self.context), None
        else:
            return None, MasterType.illegal_operation(self.pos_start, self.pos_end)
    
    def mul_by(self, other):
        if isinstance(other, Number):
            return Number(self.value * other.value).set_context(self.context), None
        else:
            return None, MasterType.illegal_operation(self.pos_start, self.pos_end)

    def exp(self, other):
        if isinstance(other, Number):
            return Number(self.value ** other.value).set_context(self.context), None
        else:
            return None, MasterType.illegal_operation(self.pos_start, self.pos_end)

    def mod(self, other):
        if isinstance(other, Number):
            return Number(self.value % other.value).set_context(self.context), None
        else:
            return None, MasterType.illegal_operation(self.pos_start, self.pos_end)
    
    def get_comparison_eq(self, other):
        if isinstance(other, Number):
            return Number(int(self.value == other.value)).set_context(self.context), None
        else:
            return None, MasterType.illegal_operation(self.pos_start, self.pos_end)
    def get_comparison_ne(self, other):
        if isinstance(other, Number):
            return Number(int(self.value != other.value)).set_context(self.context), None
        else:
            return None, MasterType.illegal_operation(self.pos_start, self.pos_end)
    def get_comparison_le(self, other):
        if isinstance(other, Number):
            return Number(int(self.value <= other.value)).set_context(self.context), None
        else:
            return None, MasterType.illegal_operation(self.pos_start, self.pos_end)
    def get_comparison_ge(self, other):
        if isinstance(other, Number):
            return Number(int(self.value >= other.value)).set_context(self.context), None
        else:
            return None, MasterType.illegal_operation(self.pos_start, self.pos_end)
    def get_comparison_lt(self, other):
        if isinstance(other, Number):
            return Number(int(self.value < other.value)).set_context(self.context), None
        else:
            return None, MasterType.illegal_operation(self.pos_start, self.pos_end)
    def get_comparison_gt(self, other):
        if isinstance(other, Number):
            return Number(int(self.value > other.value)).set_context(self.context), None
        else:
            return None, MasterType.illegal_operation(self.pos_start, self.pos_end)
    def anded_by(self, other):
        if isinstance(other, Number):
            return Number(int(self.value and other.value)).set_context(self.context), None
        else:
            return None, MasterType.illegal_operation(self.pos_start, self.pos_end)
    def ored_by(self, other):
        if isinstance(other, Number):
            return Number(int(self.value or other.value)).set_context(self.context), None
        else:
            return None, MasterType.illegal_operation(self.pos_start, self.pos_end)
    def notted(self,):
        return Number(1 if self.value ==0 else 0).set_context(self.context),None
    def is_true(self):
        return self.value != 0
        
    def copy(self):
        copy = Number(self.value)
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.context)
        return copy
    
    def __repr__(self) -> str:
        return str(self.value)

class Function(MasterType):
    def __init__(self, name, body_node, arg_names) -> None:
        super().__init__()
        self.name = name or '<anonymous>'
        self.body_node = body_node 
        self.arg_names = arg_names 
    def execute(self, args):
        res = RTResults()
        interpreter = Interpreter()
        new_context = Context(self.name, self.context, self.pos_start)
        new_context.symbol_table = SymbolTable(new_context.parent.symbol_table)

        if len(args) > len(self.arg_names):
            return res.failure(
                RTError(
                    self.pos_start, self.pos_end,self.context,
                    f"{len(args)} - {len(self.arg_names)} too many args passed into {self.name}"
                )
            )
        if len(args) < len(self.arg_names):
            return res.failure(
                RTError(
                    self.pos_start, self.pos_end,self.context,
                    f"{len(self.arg_names)} - {len(args)}   too few args passed into {self.name}"
                )
            )
        for i in range(len(args)):
            arg_name = self.arg_names[i]
            arg_value = args[i]
            arg_value.set_context(new_context)
            new_context.symbol_table.set(arg_name, arg_value)
        value = res.register(interpreter.visit(self.body_node, new_context))
        if res.error: return res 
        return res.success(value)

    def copy(self):
        copy = Function(self.name, self.body_node, self.arg_names)
        copy.set_context(self.context)
        copy.set_pos(self.pos_start, self.pos_end)
        return copy 
    
    def __repr__(self) -> str:
        return f"<function {self.name}>"

#####################
#symbol table
#####################
class SymbolTable:
    def __init__(self, parent= None) -> None:
        self.symbols = {}
        self.parent = parent 

    def get(self, name):
        value = self.symbols.get(name, None)
        if value == None and self.parent:
            return self.parent.get(name)
        return value 
    
    def set(self, name, value):
        self.symbols[name] = value 

    def remove(self, name):
        del self.symbols[name]

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
    
    def burn_VarAccessNode(self, node, context:Context):
        res = RTResults()
        var_name = node.var_name_tok.value
        value = context.symbol_table.get(var_name)
    
        if not value:
            return res.failure(
                RTError(
                    node.pos_start, node.pos_end, context, f"'{var_name}' is Undefined"
                )
            )
        value = value.copy().set_pos(node.pos_start, node.pos_end)
        return res.success(value)
    
    def burn_VarAssignedNode(self, node, context):
        res = RTResults()
        var_name = node.var_name_tok.value
        value = res.register(self.visit(node.value_node, context))
        if res.error: return res

        context.symbol_table.set(var_name, value)
        return res.success(value)


    def burn_NumberNode(self, node:NumberNode,  context:Context):
        return RTResults().success(
         Number(node.tok.value).set_context(context).set_pos(node.pos_start, node.pos_end)
        )

    def burn_BinOpNode(self, node:BinOpNode, context:Context):
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
        elif node.op_tok.type == TT_EE:
            result, error = left.get_comparison_eq(right)
        elif node.op_tok.type == TT_LE:
            result, error = left.get_comparison_le(right)
        elif node.op_tok.type == TT_GE:
            result, error = left.get_comparison_ge(right)
        elif node.op_tok.type == TT_GT:
            result, error = left.get_comparison_gt(right)
        elif node.op_tok.type == TT_LT:
            result, error = left.get_comparison_lt(right)
        elif node.op_tok.type == TT_NE:
            result, error = left.get_comparison_ne(right)
        elif node.op_tok.matches(TT_KEY, 'and'):
            result, error = left.anded_by(right)
        elif node.op_tok.matches(TT_KEY, 'or'):
            result, error = left.ored_by(right)
    
        if error:
            return res.failure(error)
        else:
            return res.success(result.set_pos(node.pos_start, node.pos_end)
                                )
    def burn_UnaryNode(self, node:UnaryNode, context:Context):
        res = RTResults()
        number = res.register(self.visit(node.node, context))
        if res.error: return res 

        error = None 
        if node.op_tok.type == TT_MINUS:
            number, error = number.mul_by(Number(-1))
        elif node.op_tok.matches(TT_KEY, 'not'):
            number, error = number.notted()

        if error: return res.failure(error)

        else:
            return res.success(number.set_pos(node.pos_start, node.pos_end))
    
    def burn_IFNode(self, node:IFNode, context:Context):
        res = RTResults()

        for condition, expr in node.cases:
            condition_value =  res.register(self.visit(condition, context))
            if res.error: return res

            if condition_value.is_true():
                expr_value =res.register(self.visit(expr, context))
                if res.error: return res 
                return res.success(expr_value)
            
            if node.else_case:
                else_value = res.register(self.visit(node.else_case, context ))
                if res.error: return res 
                return res.success(else_value)
            return res.success(None)
    def burn_ForNode(self, node: ForNode, context: Context):
        res = RTResults()
        start_value = res.register(self.visit(node.start_value_node, context))
        if res.error: return res 

        end_value = res.register(self.visit(node.end_value_node, context))
        if res.error : return res 

        if node.step_value_node:
            step_value = res.register(self.visit(node.step_value_node, context))
            if res.error : return res 
        else:
            step_value = Number(1)
        i = start_value.value
        if step_value.value >= 0:
            condition = lambda : i < end_value.value 
        else:
            condition = lambda: i > end_value.value
        
        while condition():
            context.symbol_table.set(node.var_name_tok.value, Number(i))
            i += step_value.value

            res.register(self.visit(node.body_node, context))
            if res.error: return res 
        return res.success(None)
    def burn_WhileNode(self, node:WhileNode, context:Context):
        res = RTResults()

        while True:
            condition = res.register(self.visit(node.condition_node, context))
            if res.error: return res 
            if not condition.is_true():break 

            res.register(self.visit(node.body_node, context))
            if res.error: return res 
        return res.success(None)
    def burn_FuncDefNode(self, node:FuncDefNode, context:Context):
        res = RTResults()
        func_name = node.var_name_tok.value if node.var_name_tok else None
        body_node = node.body_node
        arg_names = [arg_name.value for arg_name in node.arg_name_toks]
        func_value = Function(func_name, body_node, arg_names).set_context(context).set_pos(node.pos_start, node.pos_end)

        if node.var_name_tok:
            context.symbol_table.set(func_name, func_value)
        return res.success(func_value)
    def burn_CallNode(self, node: CallNode, context:Context):
        res = RTResults()
        args = []

        value_to_call = res.register(self.visit(node.node_to_call, context))
        if res.error:return res 
        value_to_call = value_to_call.copy().set_pos(node.pos_start, node.pos_end)

        for arg_node in node.arg_nodes:
            args.append(res.register(self.visit(arg_node, context)))
            if res.error: return res 
        return_value = res.register(value_to_call.execute(args))
        if res.error: return res 
        return res.success(return_value)
        


###################
#RUN
#################

global_symbol_table = SymbolTable()
global_symbol_table.set('null',Number(0))
global_symbol_table.set('true',Number(1))
global_symbol_table.set('false',Number(0))

def run(fn,text):
    lexer = Lexer(fn,text)
    tokens , errors = lexer.make_tokens()

    if errors: return None,errors 

    #Generate AST
    parser = Parser(tokens)
    ast = parser.parse()
    if ast.error: return None, ast.error

    context = Context('<program>')
    context.symbol_table = global_symbol_table

    interpreter = Interpreter()
    result = interpreter.visit(ast.node,context )


    return result.value, result.error