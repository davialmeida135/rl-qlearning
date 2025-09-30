# Frozen Lake + QLearning
Inicialmente, é possível perceber que o frozen lake sem estar "escorregadio" é um problema muito simples que é rapidamente aprendido pelo algoritmo.
```
slippery=False
total_episodios = 100_000
total_passos_max = 250

taxa_aprendizado = 0.1
fator_desconto = 0.99

epsilon = 1.0
epsilon_min = 0.01
taxa_decaimento = 0.00002

📊 RESULTADO FINAL: Taxa de sucesso: 100.0% (500/500)
```

Ao tornar o gelo escorregadio, a coisa fica mais complexa e o modelo só alcança 55% de sucesso.
```
slippery=True
total_episodios = 100_000
total_passos_max = 250

taxa_aprendizado = 0.1
fator_desconto = 0.99

epsilon = 1.0
epsilon_min = 0.01
taxa_decaimento = 0.00002

RESULTADO FINAL: Taxa de sucesso: 55.0% (275/500)
```
Mesmo com 1 milhão de episódios, o modelo melhorou muito pouco.
```
slippery=True

total_episodios = 1_000_000
total_passos_max = 250

taxa_aprendizado = 0.1
fator_desconto = 0.99

epsilon = 1.0
epsilon_min = 0.01
taxa_decaimento = 0.00002
RESULTADO FINAL: Taxa de sucesso: 57.6% (288/500)
```
Houve uma boa melhora ao se anular o fator de desconto
```
total_episodios = 200_000
total_passos_max = 250

taxa_aprendizado = 0.01
fator_desconto = 1

epsilon = 1.0
epsilon_min = 0.01
taxa_decaimento = 0.00002
📊 RESULTADO FINAL: Taxa de sucesso: 61.8% (309/500)
```

# Blackjack + QLearning

```
total_episodios = 100_000
total_passos_max = 250

taxa_aprendizado = 0.1
fator_desconto = 0.99

epsilon = 1.0
epsilon_min = 0.01
taxa_decaimento = 0.00002

📊 RESULTADO FINAL: Taxa de sucesso: 38.39% (19197/50000)
```

```
# 2. HIPERPARÂMETROS AJUSTADOS PARA AMBIENTE COMPLEXO
total_episodios = 100_000
total_passos_max = 250

taxa_aprendizado = 0.05
fator_desconto = 0.99

epsilon = 1.0
epsilon_min = 0.1
taxa_decaimento = 0.0000002

📊 RESULTADO FINAL: Taxa de sucesso: 38.59% (19294/50000)
```


```
# 2. HIPERPARÂMETROS AJUSTADOS PARA AMBIENTE COMPLEXO
total_episodios = 100_000
total_passos_max = 250

taxa_aprendizado = 0.05
fator_desconto = 0.99

epsilon = 1.0
epsilon_min = 0.1
taxa_decaimento = 0.002

📊 RESULTADO FINAL: Taxa de sucesso: 38.47% (19236/50000)
```

# Taxi + Q-Learning


```
fickle_passenger=False

total_episodios = 100_000
total_passos_max = 250

taxa_aprendizado = 0.01
fator_desconto = 1

epsilon = 1.0
epsilon_min = 0.01
taxa_decaimento = 0.00002

📊 RESULTADO FINAL: Taxa de sucesso: 100.0% (500/500)
```
