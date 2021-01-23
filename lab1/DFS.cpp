#include "Graph.h"
#include "MyTime.h"
#include <stack>
#include <iostream>
#include <algorithm>
using namespace std;

stack<int> road;                         //路径
bool closeList[100];                     //用于记录是否走过
int parent[100];                         //父节点

//深度搜索
bool DFS(int cur,int endId,Graph &graph){
    if(cur==endId) return true;
    else{
        for(int i=0;i<graph.getSize();i++){
            if(graph.getEdge(cur, i) != -1 && !closeList[i]){   //当前节点相邻并且在探索集合中，扩展该节点
                closeList[i]=true;                              //标记为已经访问        
                parent[i] = cur;                                //记录当前节点
                if(DFS(i,endId,graph)) return true;               
                closeList[i]=false; 
                parent[i]=-1;                                   //重置父节点
            }
        }
    }
    return false;
}

void print_result(Graph &graph,int endId)
{
    int p = endId;
    int lastNodeNum;
    road.push(p);
    while (parent[p] != -1)
    {
        road.push(parent[p]);
        p = parent[p];
    }
    lastNodeNum = road.top();
    int cost = 0;
    cout << "solution: ";
    while (!road.empty())
    {
        cout << graph.getName(road.top()) << "-> ";
        if (road.top() != lastNodeNum)
        {
            cost += graph.getEdge(lastNodeNum, road.top());
            lastNodeNum = road.top();
        }
        road.pop();
    }
    cout << "end" << endl;
    cout << "cost:" << cost<<endl;
}
int main()
{
    Graph graph;                                //图类
    graph.init();                               //信息初始化
    memset(parent, -1, sizeof(parent));
    memset(closeList, false, sizeof(closeList));
    int startId=graph.getId(startCity),endId=graph.getId(endCity);
    closeList[startId]=true;                    //表示已经访问
    myTime.Start();
    DFS(startId,endId,graph);
    myTime.End();
    double t=myTime.getTime();               
    cout<<startCity<<' '<<endCity<<endl;
    print_result(graph,endId);
    cout<<"耗时: "<<t<<"ms"<<endl;
    return 0;
}
