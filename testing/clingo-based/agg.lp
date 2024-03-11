% Layer sizes
% layer=0 - input layer
% layer(layer, size)

% Which layer is for output
% output_layer(layer)
output_layer(L) :- L = #max{ X : layer(X, _) }, layer(L, _).

% Weights
% weight(from_layer, from_node, to_node, weight)

% Biases
% bias(layer, node, bias)

% On
% on(layer, node)
% input layer
% TODO: encoding of input region
{on(0, 0..K-1)} :- layer(0, K).

% hidden layers
on(L, N) :-
    #sum{ W,I : on(L-1, I), weight(L-1, I, N, W) } >= B,
    bias(L-1, N, B), not output_layer(L).

% Output precedence
% highest order -> best candidate for output
% outpre(output, order)

% Outnode
% Sum of weights and bias
% outnode(node, sum)
outnode(N, S) :-
    S = #sum{ W,I : on(L-1, I), weight(L-1, I, N, W); B,-1 },
    output_layer(L), bias(L-1, N, B).

% Output
% output(node)
output(Node) :-
    (Sum, Order, -Node) = #max{ (S, O, -N) : outnode(N, S), outpre(N, O) }.
#show output/1.

% Target
% :- not on(2, 3). % bit on index 3 in layer 2 is on
