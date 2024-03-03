#include <incmode>.

% Layer sizes
% layer(layer, size)
output_layer(L) :- L = #max{ X : layer(X, _) }, layer(L, _).

% Input nodes
% node(layer, node)
node(L, 0..S-1) :- layer(L, S).

% Weights
% weight(from_layer, from_node, to_node, weight)

% Biases
% bias(layer, node, bias)

% On
% on(layer, node)
{on(0, N) : node(0, N)}.
on(L, N) :-
    #sum{ W,I : on(L-1, I), weight(L-1, I, N, W) } >= B,
    bias(L, N, B), not output_layer(L).

% Output precedence
% outpre(output, order)

% Outnode
% outnode(node, sum)
outnode(N, S) :-
    S = #sum{ W,I : on(L-1, I), weight(L-1, I, N, W); B,-1 : bias(L, N, B) },
    output_layer(L), bias(L, N, _).

% Output
% output(node)
output(Sum, Order, Node) :-
    (Sum, Order, Node) = #max{ (S, O, N) : outnode(N, S), outpre(N, O) }.
output(Node) :-
    output(Sum, Order, Node).

% Target
% :- not on(2, 3).

% #show on/2.
#show output/1.
#show output_layer/1.
% #show output/3.
% #show outpre/2.
% #show outnode/2.
% #show layer/2.
% #show bias/3.
