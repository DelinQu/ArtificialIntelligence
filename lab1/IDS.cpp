#include "Graph.h"
#include "MyTime.h"
#include <stack>
#include <iostream>
#include <queue>
using namespace std;
struct Node
{
    int id;                             //记录当前节点的序号
    int depth;                          //记录当前节点的层
    Node* parent;                       //父节点                       
    Node(int ii=0,int dd=0,Node* pp=NULL):id(ii),depth(dd),parent(pp){}
};
Node *root=NULL,*cur=NULL;              //创建树，用来保存IDS过程的节点信息
stack<Node*>st;                         //栈
bool closeList[100];                    //可访问集合,true表示已经访问

//迭代加深的深度优先搜索
void IDS(int startId,int endId,int maxDep,Graph &graph){
    root=new Node(startId,0,NULL); 
    closeList[startId]=true;
    st.push(root);  
    if(startId==endId) return;
    else{
        while (!st.empty())
        {
            cur=st.top();
            st.pop();
            if(cur->id==endId) return;                          //到达目标
            for(int i=0;i<graph.getSize();i++){
                if(graph.getEdge(i,cur->id)!=-1 && closeList[i]!=true && cur->depth <maxDep){  //当前节点可以访问
                    st.push(new Node(i,cur->depth+1,cur));      //将当前节点加入集合
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
    if(cur->parent==NULL){                                      //到达根节点
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
    Graph graph;                                                //图类
    graph.init();                                               //信息初始化
    memset(closeList, false, sizeof(closeList));
    int startId=graph.getId(startCity),endId=graph.getId(endCity);

    int i=0;
    myTime.Start();
    for(i=0;i<graph.getSize();i++){
        memset(closeList, false, sizeof(closeList));
        
        IDS(startId,endId,i,graph);
   
        if(cur->id==endId){
            break;
        }
    }
    myTime.End();
    double t=myTime.getTime();
    cout<<startCity<<' '<<endCity<<endl;
    cout<<"最小深度为"<<i<<endl;
    print_result(graph,cur);
    cout << "end" << endl;
    cout << "cost:" << cost<<endl;
    cout<<"耗时: "<<t<<"ms"<<endl;     
    return 0;
}