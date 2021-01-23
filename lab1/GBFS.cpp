#include "Graph.h"
#include "MyTime.h"
#include <stack>
#include <iostream>
#include <queue>
using namespace std;
struct Node
{
    int id;                         //记录当前节点的序号
    int f;                          //节点的代价函数
    Node* parent;                   //父节点                       
    Node(int ii=0,int ff=0,Node* pp=NULL):id(ii),f(ff),parent(pp){}
    //bool operator <(const Node &a)const {return f > a.f;}
};
struct cmp
{
    bool operator() (const Node *a,const Node *b)
    {
        return a->f > b->f;
    }
};
Node *root=NULL,*cur=NULL;          //创建树，用来保存IDS过程的节点信息
priority_queue<Node*,vector<Node*>,cmp>que;           
bool closeList[100];                //可访问集合,true表示已经访问

//贪婪最佳优先
void GBFS(int startId,int endId,Graph &graph){
    closeList[startId]=true;
    root=new Node(startId,graph.getH(startId),NULL); 
    que.push(root);  
    if(startId==endId) return;
    else{
        while (!que.empty())
        {
            cur=que.top();
            que.pop();
            if(cur->id==endId) return;                                      //到达目标
            for(int i=0;i<graph.getSize();i++){
                if(graph.getEdge(i,cur->id)!=-1 && closeList[i]!=true){     //当前节点可以访问
                    que.push(new Node(i,graph.getH(i),cur));                //将当前节点加入集合
                    closeList[i]=true;
                }
            }
        }
    }
    return;
}

int cost=0;
/*打印子节点*/
void print_result(Graph &graph,Node * cur)
{ 
    if(cur->parent==NULL){                      //到达根节点
        cout<<graph.getName(cur->id)<<"-> ";
        return;
    }
    //递归处理
    cost+=graph.getEdge(cur->id,(cur->parent)->id);  
    print_result(graph,cur->parent);
    cout<<graph.getName(cur->id)<<"-> ";
}
int main()
{
    Graph graph;                                //图类
    graph.init();                               //信息初始化
    memset(closeList, false, sizeof(closeList));
    int startId=graph.getId(startCity),endId=graph.getId(endCity);

    myTime.Start();
    GBFS(startId,endId,graph);               
    myTime.End();
    double t=myTime.getTime();
    cout<<startCity<<' '<<endCity<<endl;
    print_result(graph,cur);
    cout << "end" << endl;
    cout << "cost:" << cost<<endl;
    cout<<"耗时: "<<t<<"ms"<<endl;
    return 0;
}