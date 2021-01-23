#include<iostream>
#include<memory.h>
#include<map>
#include<fstream>
using namespace std;
string startCity,endCity;

/*
*created by qdl in 2020/11/23
*这是一个个人编写的图类，用于表示图的基本信息
*/


class Graph                             //图结构
{
public:
    Graph(){
        memset(graph, -1, sizeof(graph));//图初始化为-1
    }

    string getName(int id){              //获取节点名   
        return city[id];            
    }

    int getEdge(int from, int to){       //获取边
        return graph[from][to];
    }
    
    void addEdge(int from, int to, int cost){    //新增一条边
        if (from >= 20 || from < 0 || to >= 20 || to < 0)
            return;
        graph[from][to] = cost;
    }
    int getSize(){return size;}                 //获取图的规模

    int getedgeN(){return edgeN;}               //获取图的边数

    int getH(int i){return h[i];}               //获取当前节点的H估计值 

    int getId(string city){return idMap[city];}            

    void getInfo(){                             //读取文件信息
        string info;
        ifstream in("graphInfo.py");            //读取文件

        //读取城市名字
        in>>info;
        cout<<info<<endl;
        in>>size;
        for(int i=0;i<size;i++){
            in>>city[i];
            idMap[city[i]]=i;                   //建立映射
        }
        
        //读取h估计值序列
        in>>info;
        cout<<info<<endl;
        for(int i=0;i<size;i++){
            in>>h[i];
        }
        
        //读取图节点
        in>>info;
        cout<<info<<endl;
        in>>edgeN;
        for(int i=0;i<edgeN;i++){
            string u,v;
            int c;
            in>>u>>v>>c;
            addEdge(idMap[u],idMap[v],c);
        }

        //读取开始和结束城市
        in>>info;
        cout<<info<<endl;
        in>>startCity>>endCity;
        in.close();                     //关闭文件流
    }  
     
	void init(){                        //图初始化
        getInfo();                  
	}

private:
    int graph[100][100];                //图数组，用来保存图信息，最多有20个节点
    string city[100];                   //Name数组，建立id和节点名的映射
    map<string,int>idMap;               //建立名字到id的映射
    int size;                           //图的规模节点数目
    int edgeN;                          //边的数目
    int h[100];                         //从n节点到目标节点可能的最优路径的估计代价
};