#include "Graph.h"
#include "MyTime.h"
#include <stack>
#include <iostream>
#include <queue>
using namespace std;
struct Node
{
    int id;                         //记录当前节点的序号
    int level;                      //记录当前节点的层
    Node(int ii=0,int ll=0):id(ii),level(ll){}
};

queue<Node>que;                     //队列
bool closeList[100];                //可访问集合,true表示已经访问
stack<int> road;                    //路径
int levels[100]={0};                //节点记录层

//宽度优先搜索
void BFS(int startId,int endId,Graph &graph){
    closeList[startId]=true;                                    //表示已经访问
    if(startId==endId) return;
    else{
        levels[startId]=0;  
        que.push(Node(startId,0));    
        while(!que.empty()){
            Node q=que.front();                                 //取出子节点，扩展
            que.pop();
            int id=q.id,level=q.level;
            if(id==endId) return;                               //到达目标节点    
            for(int i=0;i<graph.getSize();i++){
                if(graph.getEdge(id, i) != -1 && !closeList[i]){//当前节点相邻且可访问
                    closeList[i]=true;
                    levels[i]=level+1;
                    que.push(Node(i,level+1));
                }
            }
        }
    }
    return;
}

/*打印子节点*/
void print_result(Graph &graph,int endId)
{
    int p = endId;
    int lastNodeNum;
    while (levels[p] != 0)
    {
        road.push(p);
        for(int i=0;i<graph.getSize();i++){
            if(levels[i]==(levels[p]-1) && graph.getEdge(p,i)!=-1){    //找到了父节点
                p=i;
                break;
            }
        }
    }
    road.push(p);
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
    memset(levels, -1, sizeof(levels));         //初始化levels
    memset(closeList, false, sizeof(closeList));
    int startId=graph.getId(startCity),endId=graph.getId(endCity);
    myTime.Start();
    BFS(startId,endId,graph);               
    myTime.End();
    double t=myTime.getTime();
    cout<<startCity<<' '<<endCity<<endl;
    print_result(graph,endId);
    cout<<"耗时: "<<t<<"ms"<<endl;
    return 0;
}
