---
layout: post
title: "Back Propagation and Automatic Differentiation"
author: "Kwangjin Yoon"
categories: 
  - deep__learning
tags: 
  - back propagation 
  - automatic differentiation 
  - 역전파 
  - 자동미분
---

Back propagation(BP, 백프로파게이션, 역전파)은 딥러닝에서 뉴럴넷의 weight들을 업데이트하는데 쓰이는 알고리즘이고 Automatic differentiation(AD, 자동미분)은 컴퓨터가 미분을 할수 있게 해주는 알고리즘 중 하나다. TensorFlow, PyTorch 등에서 BP가 AD를 이용하여 구현되었다고 함.

참고:
[link1](https://en.wikipedia.org/wiki/Backpropagation), 
[link2](http://neuralnetworksanddeeplearning.com/chap2.html), 
[link3](https://www.suchin.co/2017/03/18/Automatic-Differentiation-and-Backpropagation/)

-----

## Back propagation

뉴럴넷 $$N$$ 이, $$e$$ 개의 연결을 가지고 있고, $$m$$ 개의 입력을 취하며, $$n$$ 개의 출력을 가진다고하자. 즉, 어떤 트레이닝 샘플 $$(\mathbf{x}_i,\mathbf{t}_i)$$가 $$\mathbf{x}_i \in \mathbb{R}^m, \mathbf{t}_i \in \mathbb{R}^n$$ 이고, $$i=1,2,3,...$$는 트레이닝 샘플의 인덱스다. 뉴럴넷 $$N$$의 가중치 $$\mathbf{w}_j \in \mathbb{R}^e$$ 는 back propagation 알고리즘을 통해 $$j=0,1,2,...$$ 순으로 업데이트 해가며 $$\mathbf{w}_0$$ 는 초기 가중치 값이다. 다시 말하면, 첫 트레이닝 샘플 $$(\mathbf{x}_1,\mathbf{t}_1)$$과 초기 가중치 $$\mathbf{w}_0$$ 로부터 첫 번째 업데이트된 $$\mathbf{w}_1$$ 를 back propagation 과 gradient descent 알고리즘을 통해 만들어 낸다. 그리고 이 과정을 모든 트레이닝 샘플에 대해 반복한다(미니 배치 등등의 방법을 안 쓰는 경우임). 아래 그림은 $$m=3, n=2, e=20$$ 인 뉴럴넷이다.

<div style="text-align:center" markdown="1">
![An example of neural net](https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Colored_neural_network.svg/296px-Colored_neural_network.svg.png "An example of neural net")
</div>

$$i$$번째 트레이닝 샘플로부터 계산된 에러 $$E_i$$ 를 제곱차로 하자.

$$E_i = \frac{1}{2}\Vert \mathbf{t}_i-\mathbf{y}_i\Vert^2 = \frac{1}{2}\sum_{j=1}^{n}(t_{ij}-y_{ij})^2 $$

여기서 $$\mathbf{t}_i$$ 는 $$i$$ 번째 트레이닝 샘플의 정답이고, $$\mathbf{y}_i$$는 입력 $$\mathbf{x}_i$$와 뉴럴넷 $$N$$으로부터 계산된 값이다. 
굵은 글씨가 아닌 $$t_{ij}$$는 $$i$$번째 트레이닝 샘플의 정답의 $$j$$번째 차원을 말한다($$y_{ij}$$도 비슷하게 생각하면 됨). 지금부터는 특별한 언급이 없으면 트레이닝 샘플의 인덱스를 무시하고 $$E_i$$ 와 $$E$$를 구분 없이 쓰겠다($$t_{ij},y_{ij}$$도 $$t_{j},y_{j}$$ 로 출력 차원에 대한 인덱스만 사용).

이제 뉴럴넷 $$N$$의 노드에 인덱스를 부여하겠다. 위 그림을 이용할건데 제일 좌측의 input 층의 상단부터 $$1$$로 시작해서 아래 방향으로 순서를 메겨 제일 우측의 output 층의 하단까지 번호를 부여하고 그 집합을 $$\mathcal{N}$$ 이라고하자. 즉, $$\mathcal{N}=\{1,2,3,4,5,6,7,8,9\}$$. 명확히하기 위해 언급하면, input 층의 제일 하단 노드는 인덱스가 $$3$$이고 hidden 층의 제일 상단의 노드는 인덱스가 $$4$$, output 층의 제일 상단의 노드는 인덱스가 $$8$$이다. 또 output 층에 속한 노드들의 집합을 $$\mathcal{O} = \{8,9\}$$, input 층에 속한 노드들의 집합을 $$\mathcal{I}=\{1,2,3\}$$, hidden 층에 속한 노드들의 집합을 $$\mathcal{H}=\{4,5,6,7\}$$이라고 하자.

모든 노드는 이전 층의 노드들의 출력 결과와 연결된 가중치를 이용해서 자신의 출력을 내놓는다. 그럼 $$i$$번째 노드는 다음과 같이 출력 $$o_i$$를 낸다.

$$o_i = \varphi(\text{net}_i) = \varphi \left( \sum_{j}{w_{ji}o_j} \right) $$

$$\text{net}_i$$는 노드 $$i$$와 연결된 이전 층의 노드들의 output $$o_j$$와 연결된 가중치 $$w_{ji}$$의 곱들의 합이다. 여기서 $$i,j \in \mathcal{N}$$ 이다. $$w_{ji}$$ 는 노드 $$j$$ 에서 노드 $$i$$ 로의 연결에 해당하는 가중치이다. 다만, input 층에 위치한 노드의 $$o_i$$는 $$x_u$$ 이다(어떤 트레이닝 샘플의 $$u$$번째 입력 차원에 위치한 값, $$1\leq u \leq m$$). 예를들어 $$o_2$$는 $$x_2$$이다. 이번엔 $$8$$번 노드(output 층의 상단 노드)의 output $$o_8$$을 계산하면, 

$$o_8 = \varphi(\text{net}_8) = \varphi\left(\sum_{j\in\mathcal{H}}{w_{j8}o_j}\right) = y_{1} $$ 

이다. $$ y_{1} $$ 은 $$o_8$$ 가 위 그림 상에서 출력의 첫 번째 차원이라는 뜻이다. $$\varphi$$는 activiation function 이고 보통 [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) 혹은 [logistic function](https://en.wikipedia.org/wiki/Logistic_function) 등을 사용한다. 여기서는 logistic function을 사용하자. logistic function $$\varphi(z)$$은 

$$\varphi(z) = \frac{1}{1+e^{-z}}$$

이고, 이것을 미분하면

$$\frac{d\varphi}{dz}(z) = \varphi(z)(1-\varphi(z)) $$

가 된다.

그럼 에러 $$E$$를 임의의 가중치 $$w_{ji}$$에 대해서 [편미분](https://en.wikipedia.org/wiki/Partial_derivative)을 해보자. [체인룰](https://en.wikipedia.org/wiki/Chain_rule)에의해서 다음과 같이 된다.

$$\frac{\partial E}{\partial w_{ji}} = \frac{\partial E}{\partial o_{i}} \frac{\partial o_i}{\partial \text{net}_{i}} \frac{\partial \text{net}_i}{\partial w_{ji}}. $$

등호 오른쪽편 마지막 항의 $$\text{net}_i$$는 $$\sum_{k}{w_{ki}o_k}$$ 인데 이 합들 중에 한 가지($$k=j$$ 일 때)만 $$w_{ji}$$ 하고 관련이 있다. 즉,

$$ \frac{\partial \text{net}_i}{\partial w_{ji}} = \frac{\partial}{\partial w_{ji}} \left( \sum_{k}{w_{ki}o_k} \right) = \frac{\partial}{\partial w_{ji}} \left( w_{ji}o_j \right) = o_j .$$

그리고 가운데 항, $$\frac{\partial o_i}{\partial \text{net}_{i}}$$ 의 $$o_i$$는 logistic function 이므로

$$ \frac{\partial o_i}{\partial \text{net}_{i}} = \varphi(\text{net}_{i})(1-\varphi(\text{net}_{i})) = o_i(1-o_i)$$

가 된다.

첫 번째 항, $$\frac{\partial E}{\partial o_{i}}$$ 은 노드 $$i$$ 가 output 층에 위치하는지 아닌지에 따라 계산이 달라진다. 우선 노드 $$i$$가 output 층에 있다고 한다면, $$o_i = y_{u}$$가 된다. 여기서 $$u$$는 output 노드 $$i$$에 상응하는 출력 차원이다. 그러면 $$E=\frac{1}{2}\sum_j{(t_j-y_j)^2}$$ 이니까 $$o_i$$와 직접적으로 관련이 있으므로 $$o_i$$에 대해서 미분을 하면,

$$\frac{\partial E}{\partial o_i}=\frac{\partial E}{\partial y_u}=\frac{\partial}{\partial y_u}\frac{1}{2}\sum_j{(t_j-y_j)^2} = y_u - t_u$$

가 된다. 이번에는 output 층에 위치해 있지 않은 노드에 대해서 생각하자. output 층에 위치해 있지 않은 노드 $$i$$의 $$o_i$$는 $$E$$와 직접적인 관련이 없으므로 바로 첫 번째 항을 계산 할 수는 없고 바로 앞선 층에서(output 층 방향으로 한 층만 더) 계산된 미분 값을 이용해서 계산을 한다. $$o_i$$ 와 연결되어진 하나 앞선 층에 속한 노드들의 집합을 $$\mathcal{L}$$ 라고 하자($$\mathcal{L} \subset \mathcal{N}$$). 계속해서 output 층까지 해당 집합들을 찾아간다면 $$E$$를 $$o_i$$의 함수로 볼 수 있게 된다. 일단 바로 앞선 층의 $$i$$와 연결된 노드들이 $$\mathcal{L} = \{a,b,...,c\}$$ 라고 하자. 그럼 그 $$\text{net}_a,\text{net}_b,...,\text{net}_c$$ 들은 $$o_i$$를 포함하고 있고 이 $$o_a, o_b, ..., o_c$$들은 $$E$$를 향해 연결되어 있기 때문이다. 즉,

$$\frac{\partial E(o_i)}{\partial o_i} = \frac{\partial E(\varphi(\text{net}_a),\varphi(\text{net}_b),...,\varphi(\text{net}_c))}{\partial o_i} $$

으로 볼 수 있다. 그럼 $$E$$는 다변수 함수가 되고 그것의 $$o_i$$ 에 대한 [전미분](https://en.wikipedia.org/wiki/Total_derivative)은

$$\frac{\partial E}{\partial o_i} = \sum_{l\in\mathcal{L}}\left(\frac{\partial E}{\partial\text{net}_l}\frac{\partial\text{net}_l}{\partial o_i} \right) = \sum_{l\in\mathcal{L}} \left(\frac{\partial E}{\partial o_l}\frac{\partial o_l}{\partial\text{net}_l}w_{il} \right) $$

이다. 제일 우변의 $$w_{il}$$은 $$\frac{\partial\text{net}_l}{\partial o_i}$$의 결과로 나온 것이고, $$\frac{\partial E}{\partial o_l}\frac{\partial o_l}{\partial\text{net}_l}$$ 은 $$\frac{\partial E}{\partial\text{net}_l}$$의 체인룰의 결과다. 즉, $$E$$를 하나 앞선 층의 $$o_l$$에 대해서 미분하는 것을 먼저하고 뒤이어 $$o_i$$에 대한 미분을 계산하면 되는 형태로 식이 완성되게 된다.(output 층에서 input 층으로 거꾸로 하나씩 내려가면서).

끝으로 정리해보면, 에러 $$E$$를 임의의 가중치 $$w_{ji}$$에 대해서 편미분한 결과는 다음과 같다.

$$\frac{\partial E}{\partial w_{ji}} = \delta_i o_j$$

여기서 $$\delta_i$$ 는

$$\delta_i = \frac{\partial E}{\partial o_i} \frac{\partial o_i}{\partial\text{net}_i} = \left\{ \begin{array}{rcl} (y_u - t_u)o_i(1-o_i) & \text{if } i \in \mathcal{O}, \text{ and } o_i = y_u    \\ \left( \sum_{l\in\mathcal{L}}{\delta_l w_{il}} \right) o_i(1-o_i) & \text{if } i \in \mathcal{N} \setminus \mathcal{O}  \end{array} \right. $$

이다.


그럼 마지막으로 위 그림에서 가중치 $$w_{14}$$을 $$E$$에 대해서 실제로 미분을 해보고 마쳐보자. $$w_{14}$$ 에 연결된 $$4$$번 노드는 output 층에 위치해 있지 않아서 output층에 있지 않는 노드에 대해서 미분하는 방법의 좋은 예제가 될 거다. 우선 시작은 $$w_{14}$$로 $$E$$를 편미분을 하는 식을 써보자.

$$\frac{\partial E}{\partial w_{14}} =  \frac{\partial E}{\partial o_{4}} \frac{\partial o_4}{\partial \text{net}_{4}} \frac{\partial \text{net}_4}{\partial w_{14}}. $$

근데 우변의 첫 번째 항을 직접 계산하기가 힘들어 보인다. $$o_4$$가 어떤 형태로 $$E$$와 연관되어져 있는지 더 파악을 해봐야하기 때문이다. 그럼 노드 $$4$$의 집합 $$\mathcal{L}$$ 부터 찾아보면 $$o_4$$가 $$8, 9$$번 노드와 연결되어 있기 때문에 $$\mathcal{L} = \{ 8,9 \}$$ 인 것을 알 수 있다. 그러면 이 식은 다음처럼 풀 수 있다.

$$ 
\begin{eqnarray} 
\frac{\partial E}{\partial w_{14}} & = & \frac{\partial E(\varphi(\text{net}_8),\varphi(\text{net}_9))}{\partial w_{14}} \\ & = & 
\left[ \sum_{l\in\{8,9\}}\left( \frac{\partial E}{\partial\text{net}_l}\frac{\partial\text{net}_l}{\partial o_{4}} \right)  \right] \frac{\partial o_4}{\partial \text{net}_{4}} \frac{\partial \text{net}_4}{\partial w_{14}} \\ & = &
\left[ \sum_{l\in\{8,9\}}\left( \frac{\partial E}{\partial o_l}\frac{\partial o_l}{\partial \text{net}_{l}} w_{4l} \right)  \right] \frac{\partial o_4}{\partial \text{net}_{4}} \frac{\partial \text{net}_4}{\partial w_{14}}. 
\end{eqnarray} 
$$

이제 제일 마지막 줄의 가장 오른쪽 항부터 하나씩 풀어보자. 일단 $$\frac{\partial \text{net}_4}{\partial w_{14}}$$는 쉽다. $$\text{net}_4$$ 가 간단한 선형식이기때문에 $$\frac{\partial \text{net}_4}{\partial w_{14}} = o_1$$ 이 된다. 그리고 또 $$\frac{\partial o_4}{\partial \text{net}_{4}}$$ 는 위에서 계산한대로 $$\frac{\partial o_4}{\partial \text{net}_{4}} = o_4(1-o_4)$$ 가 된다. 그 다음 $$w_{4l}$$은 두 번째 줄의 $$\frac{\partial\text{net}_l}{\partial o_{4}}$$ 의 결과다. 그리고 그 앞의 두 항 $$\frac{\partial E}{\partial o_l}\frac{\partial o_l}{\partial \text{net}_{l}}$$은 두 번째 줄의 $$\frac{\partial E}{\partial\text{net}_l}$$ 의 체인룰 결과이며 위에서 정의한대로 $$\delta_l$$ 이 된다. 즉, 간단히 하면,

$$ \frac{\partial E}{\partial w_{14}} = \delta_4 o_1 = \left( \sum_{l\in\mathcal{L}}{\delta_l w_{4l}} \right) o_4(1-o_4)o_1$$

이 된다.

-----

## Automatic differentiation

컴퓨터가 미분을 하는 방법에는 Numerical differentiation, Symbolic differentiation 그리고 [Automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation)이 있다. 딥러닝에서 사용되는 에러 함수는 다양하고 activiation function도 여러가지가 쓰인다. 또한 뉴럴넷의 구조도 사용자 마음대로 구성 할 수 있다. 그렇기 때문에 그때그때 back propagation을 노드의 간선들 마다 구현해 주는 것은 (불가능은 아니겠지만) 매우 많이 번거로울 것이다. 게다가 뉴럴넷의 구조가 조금이라도 바뀌게되면 해당 부분의 BP를 다시 구현해줘야하니 엄청 비효율적일 것 같다. TensorFlow나 PyTorch는 그러한 문제점을 AD를 통해서 해결하고있기 때문에 사용자 임의대로 에러 함수를 정의하고 activiation function을 노드마다 다르게 설정 할 수도 있고 연결 구조도 마음대로 바꿀수 있는 것이다. 자기가 만든 네트워크의 BP를 어떻게 구현 할지 전혀 걱정하지 않으면서 말이다. 여기서는 AD가 무엇인지 알아보고 그것이 어떻게 BP와 연결될 수 있는지 볼 것이다. [Numerical differentiation](https://en.wikipedia.org/wiki/Numerical_differentiation)과 [Symbolic differentiation](https://en.wikipedia.org/wiki/Symbolic_differentiation)이 무엇인지는 여기서 다루지 않는다.

AD는 두 가지 모드<sup>mode</sup>가 있다. 포워드 모드<sup>forward mode</sup>와 리버스 모드<sup>reverse mode</sup>이다. 우선 포워드 모드부터 보자. AD는 `어떤 함수든 다수의 쉬운 함수들의 합성으로 나타낼수 있다`라는 점에서 착안한 방법이다. 예를들어 함수 $$f(x)=4x^2$$ 를 더 쉬운 형태의 함수 $$g(x)=x^2$$, $$h(x)=2x$$의 합성으로 $$f=g \circ h = g(h(x))=(2x)^2 = 4x^2$$ 보는 식이다. 그리고 이런 합성함수의 미분에는 체인룰<sup>chain rule</sup>이 사용된다.

$$ f(x)=g(h(x)) \Rightarrow f'(x) = g'(h(x))h'(x) $$

이제 더 어려운 함수를 예로 들어서 AD가 어떻게 함수를 미분하는지 보자. 우선 $$f(x)$$를 

$$f(x)=\frac{\ln(x)(x+3)+x^2}{\sin(x)}$$

로 정의하자. 다음은 포워드 모드에서 AD가 함수를 분해하는 방법이다.





*작성 중...*