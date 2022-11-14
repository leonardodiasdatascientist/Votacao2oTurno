# Análise das eleições para presidente do 2o turno

Datasets inclusos

Maiores informações na thread abaixo:

https://twitter.com/leonardodias/status/1592001220817870851

Caso queira rodar o notebook, pode rodá-lo em sua máquina. Para isso basta instalar o Anaconda e rodá-lo. 

https://www.anaconda.com

Também é possível instalar usando pip. Veja o site oficial do Jupyter. Tanto faz rodar o Jupyter, mais leve, ou o JupyterLab, que tem mais extensões e módulos.

https://jupyter.org/

Caso queira rodar em cloud, recomendo o Google Colab, que é bem simples, e permite que você armazene os datasets no Google Drive e acesse por lá.

https://colab.research.google.com

Também é possível criar um cluster EMR e usar o EMR Studio. Porém a criação de um cluster demanda tempo e tem custos mais altos. Mas é possível criar um para agregar ainda mais dados e aumentar o tamanho do dataset.

https://aws.amazon.com/pt/emr/features/studio/

Também será necessário instalar o pyspark, seaborn, pandas e mais alguns outros. Dependendo do pacote que você instala, alguns já vêm instalados. Mas geralmente pode ser necessário instalar o pyspark. Você pode usar os comandos abaixo:

pip install pyspark seaborn pandas
conda install pyspark

No caso do conda, pode ser necessário utilizar os pacots do conda_forge. Basta visitar a página desses projetos para verificar o comando utilizado.



