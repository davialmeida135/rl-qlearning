# Relatório
Este relatório apresenta um comparativo de resultados entre dois métodos de diferença temporal, o SARSA e o Q-Learning, em três diferentes ambientes do Gymnasium: Frozen Lake, BlackJack e Taxi.

## Algoritmos

A principal diferença entre o Q-Learning e o SARSA é o modo como ambos lidam com a política aprendida.
### Q-Learning
O Q-Learning é um algoritmo **off-policy**, que aprende sobre a política ótima independentemente da política que está sendo seguida durante o treinamento. Ele utiliza a estratégia gulosa para atualizar os valores Q, sempre considerando a ação que maximiza o valor Q no próximo estado, mesmo que esta não seja a ação efetivamente tomada pela política de exploração (como epsilon-greedy).

A implementação utilizada foi:
```python
for episodio in range(total_episodios):
    estado, info = env.reset()
    finalizado = False

    for passo in range(total_passos_max):
        if random.uniform(0, 1) < epsilon:
            acao = env.action_space.sample()
        else:
            acao = np.argmax(q_table[estado, :])

        novo_estado, recompensa, terminado, truncado, info = env.step(acao)
        finalizado = terminado or truncado

        # Atualização da Q-Table
        q_table[estado, acao] = q_table[estado, acao] + taxa_aprendizado * \
            (recompensa + fator_desconto * np.max(q_table[novo_estado, :]) - q_table[estado, acao])

        estado = novo_estado

        if finalizado:
            break

    # Atualização de Epsilon (decay)
    epsilon = max(epsilon_min, np.exp(-taxa_decaimento * episodio))
```

### SARSA
Por outro lado, o SARSA é um algoritmo **on-policy**, que aprende sobre a mesma política que está sendo utilizada para selecionar as ações durante o treinamento. Isso significa que o SARSA considera a próxima ação que será efetivamente tomada pela política atual (incluindo ações de exploração), tornando-o mais conservador em ambientes com riscos.

```python
for episodio in range(total_episodios):
    estado, info = env.reset()
    finalizado = False

    # Escolha da ação inicial
    if random.uniform(0, 1) < epsilon:
        acao = env.action_space.sample()
    else:
        acao = np.argmax(q_table[estado, :])

    for passo in range(total_passos_max):

        novo_estado, recompensa, terminado, truncado, info = env.step(acao)
        finalizado = terminado or truncado

        # Escolha da próxima ação on-policy
        if finalizado:
            proxima_acao = None
            q_proximo = 0
        else:
            if random.uniform(0, 1) < epsilon:
                proxima_acao = env.action_space.sample()
            else:
                proxima_acao = np.argmax(q_table[novo_estado, :])
            q_proximo = q_table[novo_estado, proxima_acao]

        # SARSA Update: Q(S,A) = Q(S,A) + lr[R + decay*Q(S',A') - Q(S,A)]
        q_table[estado, acao] = q_table[estado, acao] + taxa_aprendizado * \
            (recompensa + fator_desconto * q_proximo - q_table[estado, acao])

        estado = novo_estado
        acao = proxima_acao

        if finalizado:
            break

    # Atualização de Epsilon
    epsilon = max(epsilon_min, np.exp(-taxa_decaimento * episodio))
```

## Ambientes de Teste

1. **Frozen Lake**: Um ambiente de navegação em grade onde o agente deve atravessar um lago congelado evitando buracos. Este ambiente permite testar os algoritmos tanto em versões determinísticas quanto estocásticas (com superfície escorregadia).

2. **BlackJack**: Um ambiente baseado no jogo de cartas blackjack, onde o agente deve decidir quando pedir mais cartas ou parar, buscando se aproximar de 21 sem ultrapassar.

3. **Taxi**: Um ambiente onde um táxi deve navegar em uma grade, pegar passageiros e levá-los aos seus destinos, testando a capacidade dos algoritmos em tarefas hierárquicas. Também permite testar versões determinísticas e estocásticas (com mudança de destino os buscar o passageiro).


## Frozen Lake
### QLearning
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
### SARSA
```
is_slippery=False
total_episodios = 200_000
total_passos_max = 250

taxa_aprendizado = 0.01
fator_desconto = 1

epsilon = 1.0
epsilon_min = 0.01
taxa_decaimento = 0.00002
📊 RESULTADO FINAL: Taxa de sucesso: 100.0% (500/500)
```

```
is_slippery=True
total_episodios = 200_000
total_passos_max = 250

taxa_aprendizado = 0.01
fator_desconto = 1

epsilon = 1.0
epsilon_min = 0.01
taxa_decaimento = 0.00002

📊 RESULTADO FINAL: Taxa de sucesso: 49.4% (247/500)
```

Com os resultados abaixo, podemos ver que learning rates extremos não são benéficos e que o learning rate afeta muito a qualidade dos resultados.

```
is_slippery=True
total_episodios = 200_000
total_passos_max = 250

taxa_aprendizado = 0.001
fator_desconto = 0.99

epsilon = 1.0
epsilon_min = 0.01
taxa_decaimento = 0.00002
📊 RESULTADO FINAL: Taxa de sucesso: 1.8% (9/500)
```

```
is_slippery=True
total_episodios = 200_000
total_passos_max = 250

taxa_aprendizado = 0.1
fator_desconto = 1

epsilon = 1.0
epsilon_min = 0.01
taxa_decaimento = 0.00002
📊 RESULTADO FINAL: Taxa de sucesso: 13.2% (66/500)
```
## Blackjack
### QLearning
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

Teste com uma taxa de aprendizado menor e taxa de decaimento menor e epsilon mínimo maior
```
total_episodios = 100_000
total_passos_max = 250

taxa_aprendizado = 0.05
fator_desconto = 0.99

epsilon = 1.0
epsilon_min = 0.1
taxa_decaimento = 0.0000002

📊 RESULTADO FINAL: Taxa de sucesso: 38.59% (19294/50000)
```

Teste com taxa de decaimento maior
```
total_episodios = 100_000
total_passos_max = 250

taxa_aprendizado = 0.05
fator_desconto = 0.99

epsilon = 1.0
epsilon_min = 0.1
taxa_decaimento = 0.002

📊 RESULTADO FINAL: Taxa de sucesso: 38.47% (19236/50000)
```

### SARSA

```
total_episodios = 100_000
total_passos_max = 250

taxa_aprendizado = 0.1
fator_desconto = 0.99

epsilon = 1.0
epsilon_min = 0.1
taxa_decaimento = 0.00002

📊 RESULTADO FINAL: Taxa de sucesso: 38.46% (19228/50000)
Vitórias: 19228((0.38456))
Empates: 2440((0.0488))
Derrotas: 28332((0.56664))
```
![alt text](image.png)


Tentei fazer uma mudança mais radical, alterando todos os hiperparâmetros do treinamento, mas os resultados não foram diferentes.
```
total_episodios = 100_000
total_passos_max = 250

taxa_aprendizado = 0.05
fator_desconto = 0.95

epsilon = 1.0
epsilon_min = 0.01
taxa_decaimento = 0.0001
📊 RESULTADO FINAL: Taxa de sucesso: 38.57% (19286/50000)
Vitórias: 19286((0.38572))
Empates: 2405((0.0481))
Derrotas: 28309((0.56618))
```
![alt text](image-1.png)

## Taxi
### Q-Learning


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

Com fickle passenger
```
fickle_passenger=True

total_episodios = 100_000
total_passos_max = 250

taxa_aprendizado = 0.01
fator_desconto = 1

epsilon = 1.0
epsilon_min = 0.01
taxa_decaimento = 0.00002
📊 RESULTADO FINAL: Taxa de sucesso: 100.0% (500/500)
```
### SARSA



```
fickle_passenger=False

total_episodios = 200_000
total_passos_max = 250

taxa_aprendizado = 0.01
fator_desconto = 1

epsilon = 1.0
epsilon_min = 0.01
taxa_decaimento = 0.00002

📊 RESULTADO FINAL: Taxa de sucesso: 100.0% (500/500)
```

```
fickle_passenger=True
total_episodios = 200_000
total_passos_max = 250

taxa_aprendizado = 0.01
fator_desconto = 1

epsilon = 1.0
epsilon_min = 0.01
taxa_decaimento = 0.00002
📊 RESULTADO FINAL: Taxa de sucesso: 100.0% (500/500)
```