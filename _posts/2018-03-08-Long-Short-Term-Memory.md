---
layout: post
title: "Long Short-Term Memory Networks"
author: "Kwangjin Yoon"
categories: 
  - deep__learning
tags: 
  - Long Short-Term Memory
  - LSTM 
  - RNN
---

참고:
[link1](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
[link2](http://blog.varunajayasiri.com/numpy_lstm.html)

-----

## Intro to LSTM 

![a LSTM cell](http://blog.varunajayasiri.com/ml/lstm.svg "a LSTM cell")

위 그림은 LSTM의 가장 일반적인 형태(Vanilla LSTM)다. 위 그림에서 $$x_t$$ 는 입력(input)이고, $$h_{t}$$는 출력(output), $$C_{t}$$는 cell state이고 모두 벡터다. 최초 $$h_0$$와 $$C_0$$는 주어진다. (주로 영 벡터로 초기화하거나 임의의 벡터를 준다.)
LSTM은 RNN의 특별한 케이스로 RNN처럼 LSTM 셀들을 일렬로 연결해 시퀀스(혹은 시계열) 데이터를 처리 할 수 있다. 따라서 첨자 $$t$$는 $$t$$번째 시퀀스를 뜻한다. $$t$$번째 셀은 입력 $$x_t$$와 함께 이전 셀의 출력 ($$h_{t-1}$$)과 cell state ($$C_{t-1}$$)를 입력으로 받는다.

다음 그림에 3개의 LSTM cell들이 연결되어져있다.

![a LSTM network](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png "a LSTM network")

LSTM 네트워크의 핵심은 cell state인데 (위 그림에서 상단의 가로줄), 세 개의 게이트(forget gate, input gate, output gate)를 이용해서 cell state를 조절한다.
이전 셀의 cell state는 다음 셀로 입력되는데, 이때 정보를 얼마나 통과 시킬지, 또는 새로운 입력에서 어떤 정보를 취할지 등의 작업을 cell state를 통해 할 수 있다.

바로 구체적으로 파고 들어가 보자. 
![forget gate](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-f.png "forget gate")
위는 forget gate이다. $$h_{t-1}$$과 $$x_t$$를 입력으로 받아서 시그모이드 레이어로 출력하므로 출력이 0과 1 사이다. 0에 가까울수록 해당하는 정보가 희미해지게 되고 1에 가까우면 거의 손실없이 통과한다.
$$[h_{t-1},x_t]$$는 벡터 concatenation 이다. 즉, $$h_{t-1}$$이 $$N_H \times 1$$ 벡터고, $$x_t$$가 $$N_X \times 1$$ 벡터라면, $$[h_{t-1},x_t]$$의 결과는 $$(N_H+N_X)\times 1$$ 벡터다. 
그러면 $$W_f$$는 $$N_C \times (N_H+N_X) $$ 행렬이 되어야한다. $$b_f$$는 $$N_C \times 1$$ 벡터다. $$N_C$$는 cell state 벡터의 크기(원소 수)다. 시그모이드는 원소 별로(element wise) 연산을 하니까 결과적으로 $$f_t$$는 $$N_C\times 1$$의 0~1 사이의 값을 갖는 벡터다.

다음은 input gate인데 입력으로부터 어떤 정보를 얼마나 더할지 뺄지를 결정하는 역할을 한다. 이 부분은 두개의 레이어로 되어있다.
![input gate](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-i.png "input gate")
$$i_t$$는 $$N_C \times 1$$ 벡터고, $$W_i$$는 $$N_C \times (N_H+N_X)$$ 행렬이다. $$b_i$$는 $$N_C \times 1$$ 벡터다. $$W_C, b_C$$ 역시 마찬가지다. $$i_t$$는 0과 1사이의 값을 가지며 $$f_t$$와 마찬가지로 입력으로부터 정보를 얼마나 통과 시킬지를 결정한다.
$$\tilde{C}_t$$는 $$N_C \times 1$$ 벡터고, -1에서 1사이의 값을 가지며, 통과된 정보를 더할지 뺄지를 결정한다.

이제 새 cell state를 계산할 수 있다.
![cell state](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-C.png "cell state")
여기서 $$\ast$$는 원소 별(element wise) 곱셈 연산이다. 이전 cell state ($$C_{t-1}$$)에 앞서 계산한 $$f_t$$를 곱하고, $$i_t \ast \tilde{C}_t$$를 더하여 새로운 cell state를 계산한다.
당연히 $$C_t$$는 $$N_C\times 1$$ 벡터다.

마지막으로 LSTM cell의 출력을 계산한다. output gate이다.
![output gate](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-o.png "output gate")
먼저 $$N_C\times 1$$ 크기의 $$o_t$$를 계산한다. 역시 $$W_o$$는 $$N_C \times (N_H+N_X)$$ 행렬 $$b_o$$는 $$N_C \times 1$$ 벡터다. 그리고나서 $$o_t \ast \tanh (C_t)$$를 해서 출력 $$h_t$$를 계산한다.
이때 한 가지 문제가 생긴다. 일단 $$N_H=N_C$$라면 $$h_t$$의 크기가 $$h_{t-1}$$과 같기때문에 다음 LSTM 셀의 입력으로 들어 갈 수 있다. 하지만 $$N_H \neq N_C$$라면 $$h_t$$와 $$h_{t-1}$$가 다른 크기를 가져서 문제가 생긴다.
이때는 $$h_t$$의 크기를 $$N_H$$로 맞춰주는 레이어를 하나 더 둠으로써 해결한다. 이 레이어를 프로젝션(projection) 레이어라고도 부른다. 즉, $$h_t' = W_H h_t + b_H$$로 $$N_H \times 1$$ 크기를 가지는 프로젝션된 $$h_t'$$를 계산하는 레이어를 두어 $$h_t'$$를 다음 셀의 입력으로 보낸다. (이 부분은 그림에서 빠져있다. 아마 $$H=C$$인 경우에 대해서만 그린것 같다. [그림 출처](http://colah.github.io/posts/2015-08-Understanding-LSTMs/))
$$W_H$$는 $$N_H\times N_C$$ 행렬이고 $$b_H$$는 $$N_H\times 1$$ 벡터다.

LSTM 네트워크의 초기 cell state와 $$h_0$$는 영 벡터를 쓰거나 임의의 값으로 초기화 하기도 한다. 또 일련의 연결된 LSTM 셀들은 같은 웨이트를 가진다(weight sharing). 그러면 LSTM 네트워크에서 학습해야할 것들은, $$W_f, b_f, W_i, b_i, W_C, b_C, W_o, b_o$$, 그리고 프로젝션 레이어가 있다면 $$W_H, b_H$$ 이다.

-----
## LSTM in Tensorflow and Numpy

여기서는 LSTM을 numpy를 이용해 구현해보고 그 계산 결과가 tensorflow의 결과와 같은지를 확인해보자. LSTM의 forward pass만 계산하여 볼 것이다. 이것의 목적은 정말 LSTM이 우리가 이해한대로 동작하는지 확인하는 것에 있다. (텐서플로에서는 모델만 만들면 알아서 backward pass (backpropagation)를 계산해서 웨이트 업데이트까지 해준다.) 

4~7번 줄에서 입력 벡터의 크기는 3, cell state 벡터의 크기는 4, 출력 벡터의 크기는 5, 시퀀스 길이는 10으로 정했다.
8번줄의 `forget_bias_value`는 tensorflow의 LSTM에서 사용되는 변수다. 
forget gate의 bias로써 트레이닝 시작 단계의 입력이 희미해(forgetting) 지는 것을 방지하게 해준다([참고](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LSTMCell)).

10번 줄부터 31번 줄이 텐서플로로 구현한 LSTM 네트워크이다. 
11~13번 줄에서 입력($$x_t$$)과 초기 상태($$C_0, h_0$$)를 위한 placeholder를 정의해 주었고 numpy와 precision을 맞추기위해 `tf.float64`를 사용하였다.
17번 줄에서 LSTM 셀을 정의하고 있다. 출력의 크기를 `num_hidden`으로 맞춰주기위한 `num_proj=num_hidden`이 보인다.
21번 줄에서 `seq_len`만큼 연결된 LSTM 네트워크를 `dynamic_rnn` 함수로 만들었고 초기 cell state($$C_0$$)와 $$h_0$$ 벡터를 넘겨주었다.
28번 줄에서 모든 시퀀스 $$t$$에 대해서 $$x_t = (1,1,1)$$인 입력 벡터와 임의의 값을 가지는 $$C_0$$와 $$h_0$$를 LSTM 네트워크로 넘겨주고있다. 
텐서플로에서는 17번 줄처럼 `initializer`를 따로 정해주지 않을 경우, 랜덤으로 초기화된 weight를 가지는 LSTM 셀을 생성한다.
따라서 33~35번 줄에서 텐서플로로 생성된 LSTM의 weight 값들을 저장했다. 추후 이 값을 numpy로 구현한 LSTM에서 사용할 것이다.

40에서 77번 줄은 numpy로 구현한 LSTM 네트워크이다. 44~46번 줄에서 텐서플로 LSTM에서 사용했던 입력과 초기상태와 같은 값으로 numpy LSTM의 입력과 초기상태를 준비한다.
50번 줄에서 입력 `x_np[seq]`와 이전 출력 `h_np`를 concatenation하고 있다. 이 결과로 `args`의 크기(원소 수)는 `num_input + num_hidden` 인 벡터가 된다.
이제 텐서플로의 결과를 재현하기 위해서 텐서플로 LSTM의 weight 값과 bias 값을 가져와야한다. 51, 52번 줄에서 그 값들을 복사하고 있다. `weights_tf`의 0번째에는 LSTM 셀의 모든 weight들이 들어있고, 1번째에는 모든 bias들이 들어있다.
또 projection layer가 있다면 2번째에 그 weight가 들어있다 (projection layer의 경우 bias는 없다). 이때 `weights_tf`의 0번째에는 $$W_i, W_C, W_f, W_o$$가 순서대로 들어있어 $$(4N_C) \times (H+X)$$ 크기의 행렬이 들어있다.
1번째에는 그에 상응하는 bias들이 들어있어 $$(4N_C) \times 1$$ 크기의 벡터가 들어있다. 58, 59번 줄에서 이 weight와 bias들을 이용해 행렬 연산을 하는데 이때 이용되는 트릭은 다음과 같다.

$$ \underbrace { \left( \begin{array}{c} W_i \\ W_C \\ W_f \\ W_o \end{array} \right) }_{(4N_C) \times (H+X)} \cdot 
\underbrace{ \left( z_t \right)}_{(H+X)\times 1} + \underbrace{ \left( \begin{array}{c} b_i \\ b_C \\ b_f \\ b_o \end{array} \right)  }_{(4N_C) \times 1}
= \underbrace{ \left( \begin{array}{c} W_i\cdot z_t +b_i \\ W_C\cdot z_t +b_C \\ W_f\cdot z_t +b_f \\ W_o\cdot z_t +b_o \end{array} \right) }_{(4N_C) \times 1} $$

이때 $$z_t = [h_{t-1}, x_t]$$ 이다. 61번 줄에서 이 연산의 결과를 4개로 분리하고 있다. (아래 numpy 구현에서는 행렬 연산이 거꾸로인 것에 주의)

63번 줄에서 `np.tanh`로 $$\tilde{C}_t$$를 계산하고 있다.
66번 줄에서 `forget_bias`를 더하여 $$f_t$$ (forget gate)를 계산하고 있다.
68번 줄에서 $$i_t$$ (input gate)를 먼저 구하고 $$C_t$$를 계산하고 있다.
69번 줄에서 $$o_t$$ (output gate)를 먼저 구하고 출력 $$h_t$$를 계산하고 있다.
projection layer가 있을 경우 72번 줄에서 계산을 한다.
76, 77번 줄에서 계산된 `c_new` (cell state)와 `h_new` ($$h_t$$) 를 다음 시퀀스로 넘겨준다.

마지막으로 83~89번 줄에서 텐서플로 LSTM과 numpy LSTM의 forward pass 계산 결과를 비교하고 있다.

{% highlight python linenos %}
import tensorflow as tf
import numpy as np

num_input = 3
num_cell = 4
num_hidden = 5
seq_len = 10
forget_bias_value = 1

# TF LSTM
x = tf.placeholder(tf.float64, [1, seq_len, num_input]) # batch size = 1
c = tf.placeholder(tf.float64, [1, num_cell]) # batch size = 1
h = tf.placeholder(tf.float64, [1, num_hidden]) # batch size = 1

init_state = tf.contrib.rnn.LSTMStateTuple(c, h)

lstm = tf.contrib.rnn.LSTMCell(num_cell, 
    forget_bias = forget_bias_value,
    num_proj=num_hidden)

out, state = tf.nn.dynamic_rnn(lstm, x, 
    sequence_length=np.ones([1])*seq_len, # batch size = 1
    initial_state=init_state)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

out_tf, state_ = sess.run([out, state], 
    feed_dict = {x: np.array([np.ones([seq_len, num_input])]),
     c: np.array([np.linspace(0, 1, num=num_cell)]),
     h: np.array([np.linspace(0.3, 0.8, num=num_hidden)])})
     
weights_tf = []
for w_ in range(len(lstm.weights)):
    weights_tf.append( sess.run(lstm.weights[w_]) )

print('TF-LSTM')
print('\tc={}\n\th={}'.format(state_.c, state_.h))


# numpy LSTM
def sigmoid_array(x):
    return 1 / (1+np.exp(-x))

x_np = np.array( np.ones([seq_len, num_input]) )
c_np = np.array( [np.linspace(0, 1, num=num_cell)] )
h_np = np.array( [np.linspace(0.3, 0.8, num=num_hidden)] )
out_np = []

for seq in range(seq_len):
    args = np.concatenate(([x_np[seq]],h_np), axis=1)
    weights = weights_tf[0] 
    biases = weights_tf[1]

    if len(weights_tf) > 2:
        weight_proj = weights_tf[2]
        bias_proj = np.zeros([num_hidden])

    out_t = np.matmul(args, weights)
    concat = out_t + biases

    i, j, f, o = np.split(concat, 4, 1)

    c_tild = np.tanh(j)

    forget_bias = forget_bias_value
    sigmoid_f = sigmoid_array( f + forget_bias )

    c_new = sigmoid_f * c_np + sigmoid_array(i) * c_tild
    h_new = sigmoid_array(o) * np.tanh(c_new)

    if len(weights_tf) > 2:
        h_new = np.matmul(h_new, weight_proj) + bias_proj

    out_np.append( h_new )

    h_np = h_new
    c_np = c_new
    
print('NP-LSTM')
print('\tc={}\n\th={}'.format(c_new, h_new))

# check code
for chk in range(len(out_np)):
    on = out_np[chk]
    ot = out_tf[0][chk]
    diff = np.sum( np.abs(on-ot) )
    if diff > 1e-3:
        print('error')    
{% endhighlight %}

-----
## Backpropagation of LSTM

![a LSTM cell](http://blog.varunajayasiri.com/ml/lstm.svg "a LSTM cell")

*작성 중...*