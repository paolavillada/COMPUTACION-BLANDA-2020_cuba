#!/usr/bin/env python
# coding: utf-8

# In[1]:

#PRESENTADOR POR: Angie Paola villada Ortiz 1089721336 y Luz Adriana Quitumbo Santa 1088311399.

# COMPUTACIÓN BLANDA - Sistemas y Computación

# -----------------------------------------------------------------
# AJUSTES POLINOMIALES
# -----------------------------------------------------------------
# Lección 06
#
#   ** Se importan los archivos de trabajo
#   ** Se crean las variables
#   ** Se generan los modelos
#   ** Se grafican las funciones
#
# -----------------------------------------------------------------

# Se importa la librería del Sistema Operativo
# Igualmente, la librería utils y numpy

# -----------------------------------------------------------------
# -----------------------------------------------------------------
#Import OS = Nos permite acceder a funcionalidades dependientes del Sistema 
#Operativo. Sobre todo, aquellas que nos refieren información sobre el entorno del mismo y nos
#permiten manipular la estructura de directorios.
# -----------------------------------------------------------------
# -----------------------------------------------------------------

import os

# Directorios: chart y data en el directorio de trabajo
# DATA_DIR es el directorio de los datos
# CHART_DIR es el directorio de los gráficos generados

# -----------------------------------------------------------------
# -----------------------------------------------------------------
# Importamos numpy como el alias np
# -----------------------------------------------------------------
# -----------------------------------------------------------------
from utils import DATA_DIR, CHART_DIR
import numpy as np

# Se eliminan las advertencias por el uso de funciones que
# en el futuro cambiarán
# -----------------------------------------------------------------
# -----------------------------------------------------------------
#seterr se utilizan en muchos lenguajes de programación orientados 
#a objetos para garantizar el principio de encapsulación de datos.
# -----------------------------------------------------------------
# -----------------------------------------------------------------
np.seterr(all='ignore')

# Se importa la librería scipy y matplotlib
# -----------------------------------------------------------------
# -----------------------------------------------------------------
#La libreria scipy es el paquete científico (es decir, un módulo que tiene otros módulos) más completo, 
#que incluye interfases a librerías científicas muy conocidas como LAPACK, BLAS u ODR entre muchas otras.
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# Matplotlib es una biblioteca para la generación de gráficos a partir de datos contenidos en listas o arrays 
#en el lenguaje de programación Python y su extensión matemática NumPy. 
#-----------------------------------------------------------------
# -----------------------------------------------------------------

import scipy as sp
import matplotlib.pyplot as plt

# Datos de trabajo
# -----------------------------------------------------------------
# -----------------------------------------------------------------
#numpy.genfromtxt – Carga datos de un archivo de texto, con valores perdidos manejados como se especifica. 
#Función mucho más sofisticada que tiene muchos parámetros para controlar su importación.
# -----------------------------------------------------------------
# -----------------------------------------------------------------
#El método os.path.join () en Python une uno o más componentes de ruta de forma inteligente.
# -----------------------------------------------------------------
# -----------------------------------------------------------------
data = np.genfromtxt(os.path.join(DATA_DIR, "web_traffic.tsv"), 
                     delimiter="\t")

# Se establece el tipo de dato
data = np.array(data, dtype=np.float64)
print(data[:10])
print(data.shape)

# Se definen los colores
# g = green, k = black, b = blue, m = magenta, r = red
# g = verde, k = negro, b = azul, m = magenta, r = rojo
colors = ['g', 'k', 'b', 'm', 'r']

# Se definen los tipos de líneas
# los cuales serán utilizados en las gráficas
linestyles = ['-', '-.', '--', ':', '-']

# Se crea el vector x, correspondiente a la primera columna de data
# Se crea el vercot y, correspondiente a la segunda columna de data
x = data[:, 0]
y = data[:, 1]

# la función isnan(vector) devuelve un vector en el cual los TRUE
# son valores de tipo nan, y los valores FALSE son valores diferentes
# a nan. Con esta información, este vector permite realizar 
# transformaciones a otros vectores (o al mismo vector), y realizar
# operaciones como sumar el número de posiciones TRUE, con lo
# cual se calcula el total de valores tipo nan
print("Número de entradas incorrectas:", np.sum(np.isnan(y)))

# Se eliminan los datos incorrectos
# -----------------------------------------------------------------

# Los valores nan en el vector y deben eliminarse
# Para ello se crea un vector TRUE y FALSE basado en isnan
# Al negar dichos valores (~), los valores que son FALSE se vuelven
# TRUE, y se corresponden con aquellos valores que NO son nan
# Si el vector x, que contiene los valores en el eje x, se afectan
# a partir de dicho valores lógicos, se genera un nuevo vector en
# el que solos se toman aquellos que son TRUE. Por tanto, se crea
# un nuevo vector x, en el cual han desaparecido los correspondientes
# valores de y que son nan

# Esto mismo se aplica, pero sobre el vector y, lo cual hace que tanto
# x como y queden completamente sincronizados: sin valores nan
x = x[~np.isnan(y)]
y = y[~np.isnan(y)]

# CON ESTA FUNCIÓN SE DEFINE UN MODELO, EL CUAL CONTIENE 
# el comportamiento de un ajuste con base en un grado polinomial
# elegido
# -----------------------------------------------------------------
def plot_models(x, y, models, fname, mx=None, ymax=None, xmin=None):
    ''' dibujar datos de entrada '''

    # Crea una nueva figura, o activa una existente.
    # num = identificador, figsize: anchura, altura
    plt.figure(num=None, figsize=(8, 6))
    
    # Borra el espacio de la figura
    plt.clf()
    
    # Un gráfico de dispersión de y frente a x con diferentes tamaños 
    # y colores de marcador (tamaño = 10)
    plt.scatter(x, y, s=10)
    
    # Títulos de la figura
    # Título superior
    plt.title("Tráfico Web en el último mes")
    
    # Título en la base
    plt.xlabel("Tiempo")
    
    # Título lateral
    plt.ylabel("Solicitudes/Hora")
    
    # Obtiene o establece las ubicaciones de las marcas 
    # actuales y las etiquetas del eje x.
    
    # Los primeros corchetes ([]) se refieren a las marcas en x
    # Los siguientes corchetes ([]) se refieren a las etiquetas
    
    # En el primer corchete se tiene: 1*7*24 + 2*7*24 + ..., hasta
    # completar el total de puntos en el eje horizontal, según
    # el tamaño del vector x
    
    # Además, se aprovecha para calcular los valores de w, los
    # cuales se agrupan en paquetes de w*7*24. Esto permite
    # determinar los valores de w desde 1 hasta 5, indicando
    # con ello que se tiene un poco más de 4 semanas
    
    # Estos valores se utilizan en el segundo corchete para
    # escribir las etiquetas basadas en estos valores de w
    
    # Por tanto, se escriben etiquetas para w desde 1 hasta
    # 4, lo cual constituye las semanas analizadas
    plt.xticks(
        [w * 7 * 24 for w in range(10)], 
        ['semana %i' % w for w in range(10)])

    # Aquí se evalúa el tipo de modelo recibido
    # Si no se envía ninguno, no se dibuja ninguna curva de ajuste
    if models:
        
        # Si no se define ningún valor para mx (revisar el 
        # código más adelante), el valor de mx será
        # calculado con la función linspace

        # NOTA: linspace devuelve números espaciados uniformemente 
        # durante un intervalo especificado. En este caso, sobre
        # el conjunto de valores x establecido
        if mx is None:
            mx = np.linspace(0, x[-1], 1000)
        
        # La función zip () toma elementos iterables 
        # (puede ser cero o más), los agrega en una tupla y los devuelve
        
        # Aquí se realiza un ciclo .....
#-----------------------------------------------------------------
#-----------------------------------------------------------------
# En este ciclo se dice que para cada modelo y estilos se utiliza una
# paleta de colores.
# * linestyles: Los estilos de línea simples se pueden definir utilizando las cadenas 
# "sólido", "punteado", "discontinuo" o "dashdot". Se puede lograr un control más refinado 
# proporcionando una tupla de guiones .
# la funcio zip(): devuelve un iterador de tuplas basado en los objetos iterables.
# * Si no pasamos ningún parámetro, zip()devuelve un iterador vacío
# * Si se pasa un único iterable, zip()devuelve un iterador de tuplas con cada tupla 
# que tiene solo un elemento.
# * Si se pasan varios iterables, zip()devuelve un iterador de tuplas y cada tupla 
# tiene elementos de todos los iterables.
# For: es un bucle que repite el bloque de instrucciones un número prederminado de veces. 
# El bloque de instrucciones que se repite se suele llamar cuerpo del bucle y cada repetición 
# se suele llamar iteración.
        
# en este for lo que se hace es recorrer todos los modelos y estilos y se le asigna los colores
# a las listas que estan comprimidas con la funcion zip, esta funcion lo que hace es que me comprime
# estas tres listas en una sola y con el for lo que se realiza es recorrerlas. 
# con esto se obvia hacer un bucle for para cada uno de los arreglos

#-----------------------------------------------------------------
#-----------------------------------------------------------------      
        for model, style, color in zip(models, linestyles, colors):
            # print "Modelo:",model
            # print "Coeffs:",model.coeffs
            plt.plot(mx, model(mx), linestyle=style, linewidth=2, c=color)
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#legend crea una leyenda con etiquetas descriptivas para cada serie de datos trazada. 
#Para las etiquetas, la leyenda utiliza el texto de las propiedades DisplayName de la serie de datos. 
#Si la propiedad DisplayName está vacía, la leyenda utiliza una etiqueta con la forma 'dataN'. 
#La leyenda se actualiza automáticamente al agregar o eliminar series de datos de los ejes. 
#Este comando crea una leyenda para los ejes actuales o el gráfico devuelto por gca. Si los ejes actuales están vacíos, 
#entonces la leyenda está vacía. Si los ejes no existen, este comando los crea.
#-----------------------------------------------------------------
#-----------------------------------------------------------------

        plt.legend(["d=%i" % m.order for m in models], loc="upper left")

#-----------------------------------------------------------------
#-----------------------------------------------------------------
#plt.autoscale(tight=True) Es un método de conveniencia para el ajuste de escala automático de vista de eje simple. 
#Activa o desactiva el ajuste de escala automático y, luego, 
#si el ajuste de escala automático para cualquiera de los ejes está activado, 
#realiza el ajuste de escala automático en el eje o ejes especificados.
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#plt.ylim(ymin=0) Obtiene o establece los límites y de los ejes actuales.
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#if ymax:
#       plt.ylim(ymax=ymax)Establece los límites de vista del eje y.
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#if xmin:
#       plt.xlim(xmin=xmin) Obtiene o establece los límites x de los ejes actuales.
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#plt.grid(True, linestyle='-', color='0.75') Configure las líneas de la cuadrícula.
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#plt.savefig(fname) Guarde la figura actual.
#-----------------------------------------------------------------
#-----------------------------------------------------------------

    plt.autoscale(tight=True)
    plt.ylim(ymin=0)
    if ymax:
        plt.ylim(ymax=ymax)
    if xmin:
        plt.xlim(xmin=xmin)
    plt.grid(True, linestyle='-', color='0.75')
    plt.savefig(fname)

# Primera mirada a los datos
# -----------------------------------------------------------------
# -----------------------------------------------------------------
#plot_models(x, y, None, os.path.join(CHART_DIR, "1400_01_01.png"))
#crea gráficos a partir de modelos de regresión, ya sean estimaciones 
#(como los llamados gráficos de bosque o de puntos) o efectos marginales.
# -----------------------------------------------------------------
# -----------------------------------------------------------------

plot_models(x, y, None, os.path.join(CHART_DIR, "1400_01_01.png"))

# Crea y dibuja los modelos de datos
# -----------------------------------------------------------------
# -----------------------------------------------------------------
#La siguiente celda ejecuta código que genera cuatro modelos diferentes, llamados f1, f2, f3 , f10 y f100. 
#Explicaremos los nombres a continuación, pero la f significa función. 
#Cada uno de estos modelos es un tipo de función matemática ligeramente diferente. A continuación, 
#utilizaremos nuestra función de trazado para mostrar imágenes de lo que hacen los diferentes modelos.
# -----------------------------------------------------------------
# -----------------------------------------------------------------

fp1, res1, rank1, sv1, rcond1 = np.polyfit(x, y, 1, full=True)
print("Parámetros del modelo fp1: %s" % fp1)
print("Error del modelo fp1:", res1)
f1 = sp.poly1d(fp1)
# -----------------------------------------------------------------
# -----------------------------------------------------------------
## tratando de ajustar un polinomio de segundo orden
# -----------------------------------------------------------------
# -----------------------------------------------------------------
fp2, res2, rank2, sv2, rcond2 = np.polyfit(x, y, 2, full=True)
print("Parámetros del modelo fp2: %s" % fp2)
print("Error del modelo fp2:", res2)
f2 = sp.poly1d(fp2)

# -----------------------------------------------------------------
# -----------------------------------------------------------------
## tratando de ajustar un polinomio de tercer orden
# -----------------------------------------------------------------
# -----------------------------------------------------------------

f3 = sp.poly1d(np.polyfit(x, y, 3))

# -----------------------------------------------------------------
# -----------------------------------------------------------------
## tratando de encajar en un polinomio de décimo orden
# -----------------------------------------------------------------
# -----------------------------------------------------------------
f10 = sp.poly1d(np.polyfit(x, y, 10))

# -----------------------------------------------------------------
# -----------------------------------------------------------------
## tratando de ajustar un polinomio de orden centésimo
# -----------------------------------------------------------------
# -----------------------------------------------------------------
f100 = sp.poly1d(np.polyfit(x, y, 100))

# Se grafican los modelos
# -----------------------------------------------------------------
# -----------------------------------------------------------------
#
#plot_models:crea gráficos a partir de modelos de regresión, ya sean estimaciones 
#(como los llamados gráficos de bosque o de puntos) o efectos marginales.
#agregamos los modelos de las imagenes ( f1,f2,f3,f10 y f100).
# -----------------------------------------------------------------
# -----------------------------------------------------------------
plot_models(x, y, [f1], os.path.join(CHART_DIR, "1400_01_02.png"))
plot_models(x, y, [f1, f2], os.path.join(CHART_DIR, "1400_01_03.png"))
plot_models(x, y, [f1, f2, f3, f10, f100], os.path.join(CHART_DIR, "1400_01_04.png"))


# Ajusta y dibuja un modelo utilizando el conocimiento del punto
# de inflexión
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# ajustar y trazar un modelo usando el conocimiento sobre el punto de inflexión
# -----------------------------------------------------------------
# -----------------------------------------------------------------
#Hay un punto de inflexión entre la tercera y la cuarta semana.
#Entonces, según la semana 3.5, tengo que dividir los datos en dos y entrenar por separado.
# -----------------------------------------------------------------
# -----------------------------------------------------------------

inflexion = 3.5 * 7 * 24
xa = x[:int(inflexion)]
ya = y[:int(inflexion)]
xb = x[int(inflexion):]
yb = y[int(inflexion):]

# Se grafican dos líneas rectas
# -----------------------------------------------------------------
# -----------------------------------------------------------------
## ajustar dos líneas a dos conjuntos de datos
# -----------------------------------------------------------------
# -----------------------------------------------------------------

fa = sp.poly1d(np.polyfit(xa, ya, 1))
fb = sp.poly1d(np.polyfit(xb, yb, 1))

# Se presenta el modelo basado en el punto de inflexión
# -----------------------------------------------------------------
# -----------------------------------------------------------------

#crea gráficos a partir de modelos de regresión, ya sean estimaciones 
#(como los llamados gráficos de bosque o de puntos) o efectos marginales.
# -----------------------------------------------------------------
# -----------------------------------------------------------------

plot_models(x, y, [fa, fb], os.path.join(CHART_DIR, "1400_01_05.png"))

# Función de error
# ---------------------------------------------------------------------------------------------
def error(f, x, y):
    return np.sum((f(x) - y) ** 2)

#Se necesita una forma de evaluar un modelo. se resunen las diferencias 
#entre lo que predice el modelo para una hora determinada y lo que realmente sucedió a esa hora. 
#Cuanto mayor sea esta puntuación, peor será el modelo. 

# Se imprimen los errores  
# ----------------------------------------------------------------------------------------------
print("Errores para el conjunto completo de datos:")
for f in [f1, f2, f3, f10, f100]:
    print("Error d=%i: %f" % (f.order, error(f, x, y)))

print("Errores solamente después del punto de inflexión")
for f in [f1, f2, f3, f10, f100]:
    print("Error d=%i: %f" % (f.order, error(f, xb, yb)))

print("Error de inflexión=%f" % (error(fa, xa, ya) + error(fb, xb, yb)))

#Cualquier diferencia entre la predicción y la realidad 
#debería aumentar la puntuación de error; en particular, queremos que las diferencias negativas 
#(predicciones demasiado bajas) sean tan malas como las diferencias positivas 
#(predicciones demasiado altas), por lo que resumiremos las diferencias cuadradas entre 
#los valores predichos y los valores reales.

# Se extrapola de modo que se proyecten respuestas en el futuro
# ------------------------------------------------------------------------------------------------
plot_models(
    x, y, [f1, f2, f3, f10, f100],
#crea gráficos a partir de modelos de regresión, ya sean estimaciones 
#(como los llamados gráficos de bosque o de puntos) o efectos marginales

    os.path.join(CHART_DIR, "1400_01_06.png"),
    #os.path se puede usar para analizar cadenas que representan nombres de archivo 
    #en sus partes componentes.Estas funciones operan únicamente en las cadenas.

    mx=np.linspace(0 * 7 * 24, 6 * 7 * 24, 100),
    #np.linspace genera un array NumPy formado por n números equiespaciados entre dos dados
    ymax=10000, xmin=0 * 7 * 24)
#-------------------------------------------------------------------------------------------------

#Construimos funciones para esos modelos de orden 
#superior (órdenes 2, 3, 10 y 100) en las siguientes líneas de código. 

print("Entrenamiento de datos únicamente despúes del punto de inflexión")
fb1 = fb
fb2 = sp.poly1d(np.polyfit(xb, yb, 2))
fb3 = sp.poly1d(np.polyfit(xb, yb, 3))
fb10 = sp.poly1d(np.polyfit(xb, yb, 10))
fb100 = sp.poly1d(np.polyfit(xb, yb, 100))

#sp.poly1d es una clase polinomial unidimensional.(np.polyfit())Ajuste un 
#polinomio de grado deg a los puntos (x, y) .Devuelve un vector de coeficientes p 
#que minimiza el error al cuadrado en el orden deg ademas este método de clase se 
#recomienda para código nuevo ya que es más estable numéricamente.

print("Errores después del punto de inflexión")
for f in [fb1, fb2, fb3, fb10, fb100]:
    print("Error d=%i: %f" % (f.order, error(f, xb, yb)))

#Se vuelve a verificar si despues del punto de inflexión se presento algun error

# Gráficas después del punto de inflexión
# -----------------------------------------------------------------------------------------------------
plot_models(
    x, y, [fb1, fb2, fb3, fb10, fb100],
    os.path.join(CHART_DIR, "1400_01_07.png"),
    mx=np.linspace(0 * 7 * 24, 6 * 7 * 24, 100),
    ymax=10000, xmin=0 * 7 * 24)

#Vuelve y se grafican los datos dado que ya se tuvo el punto de inflexión

# Separa el entrenamiento de los datos de prueba
# ------------------------------------------------------------------------------------------------------
#Se realizara un muestreo aleatorio. Para hacer que este código siempre use la misma
# muestra aleatoria en cada ejecución, establezca la semilla aleatoria en un valor conocido.

#En esta parte se desea evaluar el rendimiento de predicción de un modelo, 
#pruébelo con algunos datos que nunca vio. Entonces pasamos un conjunto 
#de puntos para estimar el modelo: llamamos a este conjunto de puntos datos de entrenamiento . 
#Pero no dejamos que el modelo vea todos nuestros datos. 
#Tenemos algunos para la prueba, y ese conjunto se llama datos de prueba .
# Esta es una de las ideas clave del paradigma del aprendizaje automático. 
#Entrenamiento y prueba separados;no permita que el modelo en entrenamiento 
#tenga acceso a información sobre los datos de prueba mientras se está entrenando.

frac = 0.3
#valor flotante, devoluciones (valor flotante 
#longitud de los valores del marco de datos). frac no se puede utilizar con n.

split_idx = int(frac * len(xb))
shuffled = sp.random.permutation(list(range(len(xb))))
#Permutation aleatoriamente una secuencia o devolver un rango permutado.
#Si x es una matriz multidimensional, solo se baraja a lo largo de su primer índice.

test = sorted(shuffled[:split_idx])
#Sorted es una función incorporada que crea una nueva lista ordenada a partir de un iterable.

train = sorted(shuffled[split_idx:])
fbt1 = sp.poly1d(np.polyfit(xb[train], yb[train], 1))
fbt2 = sp.poly1d(np.polyfit(xb[train], yb[train], 2))
print("fbt2(x)= \n%s" % fbt2)
print("fbt2(x)-100,000= \n%s" % (fbt2-100000))
fbt3 = sp.poly1d(np.polyfit(xb[train], yb[train], 3))
fbt10 = sp.poly1d(np.polyfit(xb[train], yb[train], 10))
fbt100 = sp.poly1d(np.polyfit(xb[train], yb[train], 100))

#Estamos verificando si el modelo aprendió alguna generalización 
#útil durante el entrenamiento, generalizaciones que se aplican a datos que nunca vio, 
#o si simplemente memorizó los datos.

#------------------------------------------------------------------------------------------------------

#Se necesita una forma de evaluar un modelo. Tenemos que poder puntuar los resultados de las pruebas. 
#En este caso, es bastante fácil. Simplemente resumiremos las diferencias entre lo que predice el modelo 
#para una hora determinada y lo que realmente sucedió a esa hora. 
#Cuanto mayor sea esta puntuación, peor será el modelo. 

print("Prueba de error para después del punto de inflexión")
for f in [fbt1, fbt2, fbt3, fbt10, fbt100]:
    print("Error d=%i: %f" % (f.order, error(f, xb[test], yb[test])))

#Cualquier diferencia entre la predicción y la realidad debería aumentar la puntuación de error; 
#en particular, queremos que las diferencias negativas (predicciones demasiado bajas) 
#sean tan malas como las diferencias positivas (predicciones demasiado altas), por lo que resumiremos 
#las diferencias cuadradas entre los valores predichos y los valores reales .

#---------------------------------------------------------------------------------------------------------

#Nuevamente se crea una imagen donde estan las funciones dibujadas que el modelo debio aprender 
plot_models(
    x, y, [fbt1, fbt2, fbt3, fbt10, fbt100],
    os.path.join(CHART_DIR, "1400_01_08.png"),
    mx=np.linspace(0 * 7 * 24, 6 * 7 * 24, 100),
    ymax=10000, xmin=0 * 7 * 24)

#---------------------------------------------------------------------------------------------------------
from scipy.optimize import fsolve

#Scipy.optimize.fsolve encuentra las raíces de una función.
#Devuelve las raíces de las ecuaciones (no lineales) definidas por una estimación inicial dada.

#Se llama el metodo fsolve, este utiliza la ecuacion cuadratica y devuelve estimado de cuando se alcancen las 
#100000 solicitudes.

print(fbt2)
print(fbt2 - 100000)
alcanzado_max = fsolve(fbt2 - 100000, x0=800) / (7 * 24)
print("\n100,000 solicitudes/hora esperados en la semana %f" % 
      alcanzado_max[0])