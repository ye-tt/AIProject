import sympy

J,w = sympy.symbols('J,w')
J=w**2
##计算倒数
dJ_dw = sympy.diff(J,w)
print(dJ_dw)

###符号替换，将dJ_dw中的w 替换成2
# expr.subs([(x, 1), (y, 2)])        # 列表形式
# expr.subs({x: 1, y: 2})            # 字典形式（推荐）
dJ_dw.subs([(w,2)])
