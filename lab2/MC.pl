/*
*传道士与野人问题
*Missionaries and Cannibals solve with prolog
*/

% 状态定义 为 [CL,ML,B,CR,MR]
start([3,3,left,0,0]).
goal([0,0,right,3,3]).

% 状态是否合法
legal(CL,ML,CR,MR) :-
	ML>=0, CL>=0, MR>=0, CR>=0,
	(ML>=CL ; ML=0),	% ;连接符表示或
	(MR>=CR ; MR=0).

% 可能的移动:
% 两名传教士从左到右.
move([CL,ML,left,CR,MR],[CL,ML2,right,CR,MR2]):-
	MR2 is MR+2,
	ML2 is ML-2,
	legal(CL,ML2,CR,MR2).

% 两个食人族从左到右.
move([CL,ML,left,CR,MR],[CL2,ML,right,CR2,MR]):-
	CR2 is CR+2,
	CL2 is CL-2,
	legal(CL2,ML,CR2,MR).

% 一位传教士和一个食人族的从左到右.
move([CL,ML,left,CR,MR],[CL2,ML2,right,CR2,MR2]):-
	CR2 is CR+1,
	CL2 is CL-1,
	MR2 is MR+1,
	ML2 is ML-1,
	legal(CL2,ML2,CR2,MR2).

% 一位传教士从左到右.
move([CL,ML,left,CR,MR],[CL,ML2,right,CR,MR2]):-
	MR2 is MR+1,
	ML2 is ML-1,
	legal(CL,ML2,CR,MR2).

% 一个食人族的十字架从左到右.
move([CL,ML,left,CR,MR],[CL2,ML,right,CR2,MR]):-
	CR2 is CR+1,
	CL2 is CL-1,
	legal(CL2,ML,CR2,MR).

% 两名传教士从右到左穿过.
move([CL,ML,right,CR,MR],[CL,ML2,left,CR,MR2]):-
	MR2 is MR-2,
	ML2 is ML+2,
	legal(CL,ML2,CR,MR2).

% 两个食人族从右向左交叉.
move([CL,ML,right,CR,MR],[CL2,ML,left,CR2,MR]):-
	CR2 is CR-2,
	CL2 is CL+2,
	legal(CL2,ML,CR2,MR).

% 一位传教士和一个食人族从右向左交叉.
move([CL,ML,right,CR,MR],[CL2,ML2,left,CR2,MR2]):-
	CR2 is CR-1,
	CL2 is CL+1,
	MR2 is MR-1,
	ML2 is ML+1,
	legal(CL2,ML2,CR2,MR2).

% 一位传教士从右到左穿过.
move([CL,ML,right,CR,MR],[CL,ML2,left,CR,MR2]):-
	MR2 is MR-1,
	ML2 is ML+1,
	legal(CL,ML2,CR,MR2).

% 一个食人族从右向左交叉.
move([CL,ML,right,CR,MR],[CL2,ML,left,CR2,MR]):-
	CR2 is CR-1,
	CL2 is CL+1,
	legal(CL2,ML,CR2,MR).


% 递归调用search(state1,state2,Explored,MovesList)
search([CL1,ML1,B1,CR1,MR1],[CL2,ML2,B2,CR2,MR2],Explored,MovesList) :- 
   move([CL1,ML1,B1,CR1,MR1],[CL3,ML3,B3,CR3,MR3]), 					% 行动产生state3=[CL3,ML3,B3,CR3,MR3]
   not(member([CL3,ML3,B3,CR3,MR3],Explored)),							% 要求行动不在Explored集合中，一个剪枝操作
   % 递归调用search(state3,state2,[state3|Explored],[ [state3,state1] | MovesList ])
   search([CL3,ML3,B3,CR3,MR3],[CL2,ML2,B2,CR2,MR2],[[CL3,ML3,B3,CR3,MR3]|Explored],[[[CL3,ML3,B3,CR3,MR3],[CL1,ML1,B1,CR1,MR1]]|MovesList]).

% 找到解返回，此时扩展状态等于父状态
search([CL,ML,B,CR,MR],[CL,ML,B,CR,MR],_,MovesList):- 
	printTrace(MovesList),nl,
	writeln('推理结束'),
	length(MovesList,L),
	writeln('路径代价为':L).

% 回溯打印
printTrace([]) :- nl.
printTrace([[A,B]|MovesList]) :- 
	printTrace(MovesList), 
   	write(B), write(' --> '), writeln(A).

% 寻找传教士和食人族问题的解决方案
find:- 
	write("please input number"),nl,
	read(Num),nl,
	writeln('传教士和野人的数量为:'),
	writeln('Missionaries: 	'=Num),
	writeln('Cannibals: 	'=Num),nl,
	writeln('执行推理：'),
	search([Num,Num,left,0,0],[0,0,right,Num,Num],[[Num,Num,left,0,0]],_).
