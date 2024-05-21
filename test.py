def buscarz(matriz, valor_x):
    for fila in matriz:
        if valor_x in fila:
            return f'El valor correspondiente de z es: {fila[2]}'
        
    return 'El valor no se encuentra en la matriz'

matriz = [[0,1,2],[0,8,0], [9,0,6]]
resultado = buscarz(matriz, 9)
print(resultado)