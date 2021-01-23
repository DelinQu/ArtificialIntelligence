## 人工智能导论实验1说明文档README

《目录》：

[TOC]

- 提示，如果您没有方便阅读`README.md`文件的工具Typora，可以使用根目录下的`README.html`，用浏览器打开

## 一. 快速开始

- **运行环境**：`Ubuntu 20.04.1 LTS`

  我使用Ubuntu 20.04 LTS环境开发，在时间测试的过程中，使用了`gettimeofday`接口。这个接口专属Linux，因此，想要运行此代码，请在任意Linux环境中执行。

  

- **快速开始**：

  我为程序编写了一个简单的Makefile，用于方便地对代码进行执行管理，如果您为接下来复杂枯燥的程序说明所困扰，那么请快速开始，您可以使用如下指令编译运行或清除程序：

  `编译：`

  ```shell
  $ make
  ```

  `执行`

  ```shell
  $ make run
  ```

  `清理`

  ```shell
  $ make clean
  ```

  



## 二.代码目录结构

好了，我想您已经在`quickStart`中执行完了所有的程序，对我接下来的说明毫无兴趣了，您可以轻松的打一个低分，但是我还是要完成我的实验任务，这关乎我的实验成绩（开个玩笑）。在我的source文件夹下面，有如下10个文件，我会对他们的用途一一解释：

```shell
$ tree
.
├── AStart.cpp
├── BFS.cpp
├── DFS.cpp
├── GBFS.cpp
├── Graph.h
├── graphInfo.py
├── IDS.cpp
├── Makefile
├── MyTime.h
└── UCS.cpp
```

- `Graph.h`：图类，用于记录图的信息
- `MyTime.h`：时间测试类，用于方便的对运行程序进行时间测试
- `graphInfo.py`：图信息输入文本，也许您注意到.py，但是不要想太多，这仅仅是无聊作者我为了享受py的语法高亮加上的后缀，与py程序无关
- `BFS.cpp`：广度优先搜索
- `DFS.cpp`：深度优先搜索
- `IDS.cpp`：迭代加深搜索

- `UCS.cpp`：一致性代价搜索
- `GBFS.cpp`：贪婪最佳优先搜索
- `AStart.cpp`：A*搜索



## 三.算法说明

### （1）广度优先搜索BFS

```cpp
#include "Graph.h"
#include "MyTime.h"
#include <iostream>
#include <queue>
using namespace std;
struct Node{}						//节点

queue<Node>que;                     //队列
bool closeList[100];                //可访问集合,true表示已经访问
stack<int> road;                    //路径
int levels[100]={0};                //节点记录层

//宽度优先搜索
void BFS(int startId,int endId,Graph &graph){}

/*打印子节点*/
void print_result(Graph &graph,int endId){}

int main(){}
```

- 程序说明

程序BFS.cpp主要包含了4个模块，其中`struct Node{}`定义了搜索过程的节点，`void BFS(int startId,int endId,Graph &graph){}` 是我实现BFS算法主体，`void print_result(Graph &graph,int endId){}`用于根据搜索的结果打印一条路径。`mian()`则是程序执行入口。

- 注意

这条程序可以直接在Linux环境中执行，但是程序依赖一些头文件和图文本，请确保路径中可以找到 `"Graph.h"`和 `"MyTime.h"`两个头文件和`graphInfo`图文本。

### （2）深度优先搜索DFS

```cpp
#include "Graph.h"
#include "MyTime.h"
#include <stack>
#include <iostream>
#include <algorithm>
using namespace std;

stack<int> road;                         		//路径
bool closeList[100];                     		//用于记录是否走过
int parent[100];                         		//父节点


bool DFS(int cur,int endId,Graph &graph){}		//深度优先搜索

void print_result(Graph &graph,int endId){}		//打印

int main(){}									//主函数

```

- 程序说明

程序DFS.cpp主要包含了4个模块，其中`stack<int>road`定义了搜索的路径，`void DFS(int startId,int endId,Graph &graph){}` 是我实现DFS算法主体，`void print_result(Graph &graph,int endId){}`用于根据搜索的结果打印一条路径。`mian()`则是程序执行入口。

- 注意

这条程序可以直接在Linux环境中执行，但是程序依赖一些头文件和图文本，请确保路径中可以找到 `"Graph.h"`和 `"MyTime.h"`两个头文件和`graphInfo`图文本。



### （3）迭代加深搜索IDS

```cpp
#include "Graph.h"
#include "MyTime.h"
#include <stack>
#include <iostream>
#include <queue>
using namespace std;
struct Node{}							//记录节点
Node *root=NULL,*cur=NULL;              //创建树，用来保存IDS过程的节点信息
stack<Node*>st;                         //栈
bool closeList[100];                    //可访问集合,true表示已经访问


void IDS(int startId,int endId,int maxDep,Graph &graph){}	//迭代加深的深度优先搜索

void print_result(Graph &graph,Node * cur){}				//打印子节点

int main(){}
```

- 程序说明

程序DFS.cpp主要包含了4个模块，其中`Node`定义了搜索的节点，`void IDS(int startId,int endId,Graph &graph){}` 是我实现IDFS算法主体，`void print_result(Graph &graph,int endId){}`用于根据搜索的结果打印一条路径。`mian()`则是程序执行入口。

- 注意

这条程序可以直接在Linux环境中执行，但是程序依赖一些头文件和图文本，请确保路径中可以找到 `"Graph.h"`和 `"MyTime.h"`两个头文件和`graphInfo`图文本。



### （4）一致性代价搜索UCS

```cpp
#include "Graph.h"
#include "MyTime.h"
#include <stack>
#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;

struct node{};                         	//一个节点结构，node

bool list[100];                         //用于记录是否走过
vector<node> frontier;                  //扩展节点集合
bool explored[100];                     //可访问集合
stack<int> road;                        //路径
int parent[100];                        //父节点

void UCS(int endCity,node &src,Graph &graph){}

void print_result(Graph &graph,int endId){}

int main(){}
```

- 程序说明

程序UCS.cpp主要包含了4个模块，其中`Node`定义了搜索的节点，`void UCS(int startId,int endId,Graph &graph){}` 是我实现UCS算法主体，`void print_result(Graph &graph,int endId){}`用于根据搜索的结果打印一条路径。`mian()`则是程序执行入口。

- 注意

这条程序可以直接在Linux环境中执行，但是程序依赖一些头文件和图文本，请确保路径中可以找到 `"Graph.h"`和 `"MyTime.h"`两个头文件和`graphInfo`图文本。



### （5）贪婪最佳优先搜索GBFS

```cpp
#include "Graph.h"
#include "MyTime.h"
#include <stack>
#include <iostream>
#include <queue>
using namespace std;
struct Node{};										//搜索节点

struct cmp{};										//排序用到的结构

Node *root=NULL,*cur=NULL;          				//创建树，用来保存IDS过程的节点信息
priority_queue<Node*,vector<Node*>,cmp>que;           
bool closeList[100];                				//可访问集合,true表示已经访问


void GBFS(int startId,int endId,Graph &graph){}		//贪婪最佳优先

int cost=0;

void print_result(Graph &graph,Node * cur){}		//打印子节点
int main(){}
```

- 程序说明

程序GBFS.cpp主要包含了5个模块，其中`Node`定义了搜索的节点，`struct cmp{};`则是我排序用到的结构，`void GBFS(int startId,int endId,Graph &graph){}` 是我实现GBFS算法主体，`void print_result(Graph &graph,int endId){}`用于根据搜索的结果打印一条路径。`mian()`则是程序执行入口。

- 注意

这条程序可以直接在Linux环境中执行，但是程序依赖一些头文件和图文本，请确保路径中可以找到 `"Graph.h"`和 `"MyTime.h"`两个头文件和`graphInfo`图文本。

### （6）A*搜索AStart

```cpp
#include "Graph.h"
#include "MyTime.h"
#include <stack>
#include <vector>
#include <iostream>
#include <algorithm>
using namespace std;
struct node{};                           //一个节点结构，node

bool list[100];                          //用于记录是否走过
vector<node> openList;                   //扩展节点集合
bool closeList[100];                     //可访问集合
stack<int> road;                         //路径
int parent[100];                         //父节点
void A_star(int endId,node &src,Graph &graph){}

void print_result(Graph &graph,int endId){}

int main(){}
```

- 程序说明

程序`UCS.cpp`主要包含了4个模块，其中`node`定义了搜索的节点，`void A_star(int endId,node &src,Graph &graph){}` 是我实现A*算法主体，`void print_result(Graph &graph,int endId){}`用于根据搜索的结果打印一条路径。`mian()`则是程序执行入口。

- 注意

这条程序可以直接在Linux环境中执行，但是程序依赖一些头文件和图文本，请确保路径中可以找到 `"Graph.h"`和 `"MyTime.h"`两个头文件和`graphInfo`图文本。

