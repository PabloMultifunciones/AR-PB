# AR-PB
Aprendizaje Reforzado - Policy Based
### Introduccion ###

En DQN, tomamos medidas con el valor Q más alto (recompensa futura máxima esperada en cada estado) para elegir qué acción tomar en cada estado. Por lo tanto, las políticas de aprendizaje basadas en valores existen solo debido a estas evaluaciones del valor de la acción. En este tutorial, aprenderemos una técnica de aprendizaje por refuerzo basada en políticas llamada Gradientes de políticas.  

Ya dije que el objetivo principal del agente de aprendizaje por refuerzo es aprender alguna función de política π que asigna el espacio de estado S al espacio de acción A. Con DQN y con cualquier otro agente de optimización de valor, π se aprende indirectamente al estimar una función de valor tal como el valor Q óptimo. Con los agentes de optimización de políticas, π se aprende directamente.  

Esto significa que estamos tratando directamente de optimizar nuestra función de política π sin preocuparnos por la función de valor. Ahora parametrizaremos π (seleccione una acción sin una función de valor).  

Los algoritmos de gradiente de política (PG), que pueden realizar un ascenso de gradiente en π directamente, ejemplifican un algoritmo de aprendizaje de refuerzo particularmente conocido llamado REINFORCE. La ventaja de los algoritmos PG como REINFORCE es que es probable que converjan con la solución óptima, por lo que se usan más ampliamente que los algoritmos de optimización de valor como DQN.  

La compensación es que PG tiene baja consistencia. Tienen una diferencia más significativa en su rendimiento en comparación con las técnicas de optimización de valor como DQN, por lo que los PG generalmente requieren una cantidad más significativa de muestras de entrenamiento.  

### ¿Por qué utilizar métodos basados en políticas? ###

Primero, definiremos una red de políticas que implemente, por ejemplo, conductores de automóviles de IA. Esta red tomará el estado del automóvil (por ejemplo, la velocidad del automóvil, la distancia entre el automóvil y las líneas de la pista) y elegirá lo que debemos hacer (girar a la izquierda o a la derecha, pisar el pedal de velocidad o pisar el freno) . Se conoce como Policy-Based Reinforcement Learning porque, como resultado, vamos a parametrizar directamente la política. Así es como se verá nuestra función de política:  

![download](https://user-images.githubusercontent.com/95035101/201487417-a4996015-556d-4e3e-b5ac-336cf6ae6da3.svg)


s - valor de estado;   
a - valor de acción;  
Q - parámetros del modelo de la red de políticas. Podemos pensar que la política es el comportamiento del agente, es decir, una función para mapear de estado a acción.  

Una política puede ser determinista o estocástica. Una política determinista es una política que asigna estados a acciones. Damos un estado actual a la política, y la función devuelve una acción que debemos tomar:  

* Las políticas deterministas se utilizan en entornos deterministas. Son entornos donde la acción realizada verifica el resultado. No hay incertidumbre. Por ejemplo, cuando juegas al ajedrez y mueves tu peón de A3 a A4, estás seguro de que tu peón se moverá a A4.
Política determinista: a=μ(s)  
Por otro lado, una política estocástica proporciona una distribución de probabilidad para las acciones.  
* La política estocástica significa que en lugar de estar seguros de actuar a (por ejemplo, a la izquierda), existe la probabilidad de que tomemos una diferente (en este caso, 30% de que tomemos a la derecha). La política estocástica se usa principalmente cuando el entorno es incierto. Este proceso se denomina Proceso de decisión de Markov parcialmente observable (POMDP).  
La mayoría de las veces, usaremos este segundo tipo de política. Política estocástica: π(a|s)=P[a|s]  

### Ventajas de usar gradientes de política ###

Los métodos basados en políticas tienen mejores características de convergencia. El problema con las estrategias basadas en valores es que pueden tener una oscilación importante durante el entrenamiento. Esto se debe a que la elección de la acción puede cambiar drásticamente por un cambio arbitrariamente pequeño en los valores de acción estimados.  

Por el contrario, con un gradiente de política, tendemos a seguir la pendiente para encontrar los mejores parámetros. Tenemos una actualización fluida de nuestra política en cada paso.  

Debido a que tendemos a seguir el gradiente para buscar los mejores parámetros, tenemos la garantía de coincidir con el máximo local (el peor de los casos) o el máximo global (el mejor punto):  

![local_maximum](https://user-images.githubusercontent.com/95035101/201487562-9ada5f8f-24a2-4172-b04b-2e5e3c564ba5.png)

La segunda ventaja es que los gradientes de políticas son más efectivos en espacios de acción a gran escala o utilizando acciones continuas.  

El problema con Deep Q-learning es que sus predicciones asignan una puntuación (recompensa futura máxima esperada) para cada acción posible, dentro de cada paso de tiempo, dado el estado actual.  

Pero, ¿y si tenemos infinitas posibilidades de acción? Tomemos el mismo ejemplo, con un coche autónomo, en cada estado, y podemos tener una (casi) infinita elección de actividades (girar el volante a 15°, 20°, 24°, …). ¡Tendremos que proporcionar un valor Q para cada acción posible!  

Por otro lado, en los métodos basados en políticas, modificamos los parámetros directamente: el agente comenzará a comprender cuál será el máximo, en lugar de calcular (estimar) el máximo directamente en cada paso.  

![PG_vs_DQN](https://user-images.githubusercontent.com/95035101/201487586-1e72a003-b745-4396-8ab5-bdd30b7aa2d2.png)

Una tercera ventaja es que el gradiente de política puede aprender una política aleatoria, mientras que las funciones de valor no pueden. Esto tiene dos consecuencias.  

Uno de ellos es que no tenemos que comprometernos e implementar una compensación de explotación/exploración. Una política aleatoria le permite a nuestro agente explorar el espacio de estado sin realizar continuamente la misma acción. Esto se debe a que genera una distribución de probabilidad sobre diferentes acciones. En consecuencia, maneja la compensación de exploración/explotación sin codificarla de forma rígida.  

Además, nos deshacemos del problema del aliasing perceptivo. El alias perceptual es cuando tenemos dos estados que parecen ser (o son) iguales pero necesitan acciones completamente diferentes. Por ejemplo, considere el juego piedra, papel o tijera. Una política determinista, por ejemplo, jugar solo a las tijeras, podría explotarse fácilmente, por lo que un enfoque aleatorio tiende a funcionar mucho mejor.  

Tomemos el siguiente ejemplo. Tenemos un buscador de oro inteligente cuyo objetivo es encontrar el oro y evitar que lo maten.  

![Example_1](https://user-images.githubusercontent.com/95035101/201487611-fd433c2e-9887-40d8-b502-3c434e19859e.png)

El alias perceptual es cuando nuestro agente no puede diferenciar la mejor acción a realizar en un estado que se ve muy similar. Nuestro agente ve estos cuadrados gris oscuro como estados idénticos. Esto significa que un agente determinista tomaría la misma acción en ambos estados.  

Supongamos que el objetivo de nuestro agente es llegar al tesoro evitando el fuego. Los dos estados grises oscuros tienen un alias perceptivo; en otras palabras, el agente no puede ver la diferencia entre ellos porque parecen idénticos. En el caso de una política determinista, el agente haría la misma acción para ambos estados y nunca llegaría al tesoro. Sin embargo, una política aleatoria puede moverse hacia la izquierda o hacia la derecha, lo que le da una mayor probabilidad de alcanzar el premio. La única esperanza es realizar la acción inesperada seleccionada por la técnica de exploración ávida de épsilon.  

Al igual que con los procesos de decisión de Markov, una desventaja de las estrategias basadas en políticas es que normalmente tardan más en converger y evaluar la política requiere más tiempo. Otra desventaja es que tienden a converger al óptimo local en lugar del óptimo global.  

A pesar de estos escollos, los gradientes de políticas tienden a funcionar mejor que los agentes de aprendizaje por refuerzo basados en valores en tareas complejas. Como veremos en tutoriales futuros, varios avances en el aprendizaje por refuerzo al vencer a los humanos en juegos avanzados como DOTA usan técnicas basadas en gradientes de políticas.  

![Example_2](https://user-images.githubusercontent.com/95035101/201487638-8f386067-f4c5-44e6-873c-8f779fc31bb6.png)

### Desventajas ###

Naturalmente, los gradientes de política tienen una gran deficiencia. Durante mucho tiempo convergen con el máximo local, no con el óptimo global. Otra desventaja de los métodos basados en políticas es que, por lo general, tardan más en converger y evaluar la política lleva mucho tiempo.  

En lugar de Deep Q-Learning, siempre tratando de alcanzar el máximo, los pasos del gradiente de políticas convergen gradualmente. Pueden tomar más tiempo para entrenar.  

Sin embargo, veremos que hay soluciones a este problema. Ya mencioné que tenemos nuestra política π que tiene un parámetro Q. Este π genera una distribución de probabilidad de actuar en un estado dado S con parámetros theta:  

![download](https://user-images.githubusercontent.com/95035101/201487699-0e91c637-5314-4e07-a5f1-12f9765aab98.svg)

Quizás se pregunte, ¿cómo sabemos si nuestra política es lo suficientemente buena? Tenga en cuenta que la política puede verse como un problema de optimización. Necesitamos encontrar los mejores parámetros (Q) para maximizar la función de puntuación, J(Q). Hay dos pasos:  

* Podemos medir la calidad de una política con una función de puntaje de política J(Q);  
* Podemos usar el ascenso de gradiente de política para encontrar el mejor parámetro Q que mejore nuestra política.  

La idea básica es que J (Q) nos dirá qué tan buena es nuestra política (π). El ascenso del gradiente de políticas nos ayudará a encontrar los mejores parámetros de políticas para maximizar el ejemplo de buena acción.  

Independientemente de estos escollos, los gradientes de políticas funcionan mejor que los agentes de aprendizaje por refuerzo basados en valores en tareas complejas. De hecho, muchos de los avances en el aprendizaje por refuerzo superando a los humanos en juegos complejos como DOTA utilizan técnicas basadas en gradientes de políticas, como veremos en los próximos tutoriales.  

### Función de puntaje de política J(Q) ###

Tres métodos funcionan igualmente bien para optimizar las políticas. La elección depende únicamente del entorno y de los objetivos que tengamos. Para medir qué tan buena es nuestra política, usamos una función llamada función objetivo o función de puntaje de la política que calcula la recompensa esperada de la política. En un entorno episódico, siempre usamos el valor inicial. Luego, calculamos la media del retorno del primer paso de tiempo (G1), que se denomina recompensa acumulada con descuento para todo el episodio:  

![download](https://user-images.githubusercontent.com/95035101/201487774-7119b3d0-1e66-404f-bffc-d55d74d2184e.svg)

- [G1=R1+γ+γ2R3+...] - recompensa acumulada con descuento a partir del estado de inicio; Eπ(V(s1)) - el valor del estado 1.

La idea es simple, si siempre comenzamos en algún estado s1, ¿qué recompensa total obtendremos desde ese estado inicial hasta el final? Perder, al principio, está bien, pero queremos mejorar el resultado. Por ejemplo, si jugamos un nuevo juego de Pong, perdemos la pelota en la primera ronda. Una nueva ronda siempre comienza en el mismo estado. Calculamos el resultado usando J1 (Q). Para hacer esto, necesitaremos refinar la distribución de probabilidad de mis acciones ajustando los parámetros.  

Podemos usar el valor promedio en un entorno continuo porque no podemos confiar en un estado inicial específico. Debido a que algunos estados ocurren más que otros, cada valor de estado debe ser ponderado por la probabilidad de ocurrencia del estado respectivo:  

![download](https://user-images.githubusercontent.com/95035101/201487787-7cea7eb7-1501-4558-ac01-cadb5c4d5a85.svg)

Finalmente, podemos usar la recompensa promedio por paso. La idea es que queremos obtener la máxima bonificación para cada etapa:

![download](https://user-images.githubusercontent.com/95035101/201487797-d7808a53-dab6-40b2-86ac-a4521b3a1c41.svg)

- Número de ocurrencias del estado; ∑s'N(s') Número total de ocurrencias de todos los estados.  

Finalmente, podemos usar la recompensa promedio por paso de tiempo. La idea aquí es que queremos obtener la mayor recompensa por paso de tiempo:  

![download](https://user-images.githubusercontent.com/95035101/201487825-9b6ee570-bfb3-4d02-90fc-bf9d75588482.svg)

### Ascenso de gradiente de política ###

Tenemos una función de calificación de políticas que nos dice qué tan buenas son nuestras políticas. Ahora queremos encontrar el parámetro Q que pueda maximizar esta función de puntuación. Maximizar la función de puntaje significa encontrar la política óptima.  

Para maximizar la función de puntaje, necesitamos usar el llamado ascenso de gradiente en los parámetros de política.  

Política El ascenso de gradiente es el inverso del descenso de gradiente (el descenso de gradiente siempre muestra el cambio más pronunciado). Descendiendo el gradiente, tomamos la dirección de la disminución más pronunciada de la función. A medida que ascendemos por la pendiente, tomamos el camino más rápido de aumento de la función.  

¿Por qué gradiente ascendente y no gradiente descendente? Como resultado, tendemos a usar el descenso de gradiente después de tener una función de error que deseamos reducir.  

Pero tenga en cuenta que la función de puntuación no es una función de error. Es una función de puntaje, y dado que deseamos maximizar el puntaje, queremos usar el ascenso de gradiente.  

La idea principal es encontrar el gradiente de la política actual, que actualiza los parámetros en la dirección del aumento más significativo y se repite.  

Implementemos matemáticamente el ascenso de gradiente de políticas. Este tutorial es desafiante. Sin embargo, es fundamental entender cómo llegar a nuestra fórmula de gradiente.  

Entonces, queremos encontrar los mejores parámetros Q* que maximicen la puntuación:  

![download](https://user-images.githubusercontent.com/95035101/201487872-6e5d7c9b-6b3c-4ba0-860d-1f5572a3a44b.svg)

Aquí, los argumentos que están después de argmax, a partir de E, son iguales a J(Q), por lo que nuestra función de puntuación se puede definir como:  

![download](https://user-images.githubusercontent.com/95035101/201487881-b00eae01-91f8-4cfa-9e0a-56889b0c93f1.svg)

E - política dada esperada; τ - Recompensa futura esperada.  

La fórmula anterior nos muestra la suma total de la recompensa esperada con una política dada. Ahora, necesitamos hacer un ascenso de gradiente para diferenciar nuestra función de puntuación J(Q). Nuestra función de puntuación J(Q) también se puede definir como:  

![download](https://user-images.githubusercontent.com/95035101/201487952-28421f08-8195-4dba-91b7-2ea307a375ba.svg)

Escribí la función de esta manera para mostrar el problema que enfrentamos aquí. Sabemos que los parámetros de las políticas cambian la forma en que se eligen las acciones y, como resultado, recibimos las recompensas que verán los estados y con qué frecuencia.  

Por lo tanto, no será muy fácil buscar cambios de política para garantizar la mejora. Esto se debe a que el rendimiento depende de las elecciones de acciones y la distribución de los estados en los que se realizan esas elecciones.  

Ambos se ven afectados por los parámetros de la política. El impacto de los parámetros de política sobre las acciones es fácil de encontrar, pero ¿cómo encontrar el impacto de la política sobre la distribución del estado? La función de entorno es desconocida. Como resultado, nos enfrentamos a un problema: ¿cómo estimar ∇ (gradiente) con respecto a la política Q cuando el gradiente depende del efecto desconocido de los cambios de política en la distribución del estado?  

La solución al problema anterior es utilizar el teorema del gradiente de política. Esto proporciona una expresión analítica para el gradiente J (Q) (actividad) en relación con la política Q, que no implica una diferenciación de la distribución estatal.  

Entonces, continuando con el cálculo de la expresión J(Q) anterior, podemos escribir:  

![download](https://user-images.githubusercontent.com/95035101/201487998-4070782c-57b2-4280-ab2b-19c93eb32086.svg)

Ahora, estamos en una situación de política estocástica. Pondremos el gradiente dentro de la fórmula anterior en lugar de todo hasta R(τ). Esto significa que nuestro enfoque proporciona una distribución de probabilidad π(τ; Q). Dados nuestros parámetros actuales Q, se obtiene la probabilidad de realizar estos pasos (s0, a0, r0,…).  

Pero no es fácil separar la función de probabilidad a menos que podamos convertirla en un logaritmo. Esto hace que sea mucho más fácil de distinguir. Por lo tanto, usaremos un truco de razón de probabilidad que convertirá la parte resultante en probabilidad logarítmica. El truco de la razón de probabilidad se ve así:  

![download](https://user-images.githubusercontent.com/95035101/201488017-a04696f0-ca5f-4bde-8d5b-840467f7089f.svg)

De la expresión anterior, podemos escribir:

![download](https://user-images.githubusercontent.com/95035101/201488032-fd33d2af-ec3c-4958-8518-a2aba9827144.svg)

Con la expresión anterior, podemos obtener la siguiente expresión:  

![download](https://user-images.githubusercontent.com/95035101/201488051-d8f8a84f-8e31-4c53-b668-d5c0198ae012.svg)

Finalmente, podemos convertir la suma de nuevo a la expectativa:  

![download](https://user-images.githubusercontent.com/95035101/201488069-3bea7b2e-d5b6-44eb-96ff-a85b198de969.svg)

π(τ|Q) - Función de política; R(τ) - Función de puntuación.  

Como puede ver, solo necesitamos calcular la derivada de la función de política de registro. Ahora que hemos hecho eso, y fue mucho, podemos concluir sobre los gradientes de políticas:  

![download](https://user-images.githubusercontent.com/95035101/201488105-104594bc-d06a-4589-8504-481e1d2e2d26.svg)

π(s,a,Q) - Función política; R(τ) - Función de puntuación.  

Y nuestra regla de actualización se ve así:  

![download](https://user-images.githubusercontent.com/95035101/201488113-36e77faf-9653-41e5-a0a5-9b4c3718c879.svg)

△Q - Cambio en los parámetros; α - Tasa de aprendizaje.  

Este gradiente de política nos dice cómo debemos cambiar la distribución de la política cambiando los parámetros Q para lograr una puntuación más alta.

R(τ) es como una puntuación de valor escalar:

* Si R(τ) es alto, significa que tomamos medidas que generaron recompensas altas en promedio. Queremos acelerar la probabilidad de acciones visibles (aumentar la posibilidad de que se realicen estas acciones).  

* Por otro lado, si R (τ) es pequeño, queremos desplazar la probabilidad de acción vista.  

En términos simples, la política es la mentalidad del agente. El agente ve la situación actual (estado) y elige elegir una acción. (o acciones múltiples), por lo que definimos la política como una función del estado que genera algunas acciones: acción = f (estado) llamamos a la política de gradiente de política de la función f anterior.  








