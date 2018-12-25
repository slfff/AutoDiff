classdef vector < handle
    
%%%%%%%%%%%%%%% Lingfei Song 2018.12.13 %%%%%%%%%%%%%%%%%%
% This class implements the fundamental data structure of 
% our autodiff model. All the variables should be converted 
% to our vector class, using obj = vector(var). The constructor 
% accepts any dimension input (scalar, vector, matrix, ...), 
% and internally they are all treated as a vector. 
% value: the raw variable
% dydx: the derivative of next node(s) against this node
% pre: previous node(s)
% next: next node(s) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    properties
        value;
        dydx;   % for reverse mode
                
                % dy1/dx1 dy2/dx1 ... dym/dx1
                % dy1/dx2 dy2/dx2 ... dym/dx2
                % ...     ...     ... ...
                % dy1/dxn dy2/dxn ... dym/dxn
                
        pre;    % previous node(s)
        next;   % next node(s)
        dim;    % dimsion of each next node
        flag;   % denote whether the next node (i) has been processed
    end
    methods
        function obj = vector(val)
            obj.value = val;
            obj.dydx = [];
            obj.pre = [];
            obj.next = [];
            obj.dim = [];
            obj.flag = [];
        end
        function y = plus(x1, x2)
            assert(isa(x1, 'vector')|isa(x2, 'vector'));
            if isa(x1, 'vector') && isa(x2, 'vector')
                y = vector(x1.value + x2.value);
                x1.dydx = [x1.dydx, sparse(1:numel(x1.value), 1:numel(y.value), ones(numel(x1.value),1))]; 
                x2.dydx = [x2.dydx, sparse(1:numel(x1.value), 1:numel(y.value), ones(numel(x1.value),1))];
                y.pre = [x1, x2];
                x1.next = [x1.next, y];
                x2.next = [x2.next, y];     % structure vector
                x1.dim = [x1.dim, numel(y.value)];
                x2.dim = [x2.dim, numel(y.value)]; 
                x1.flag = [x1.flag, 0];
                x2.flag = [x2.flag, 0];
            elseif isa(x1, 'vector')
                y = vector(x1.value + x2);
                x1.dydx = [x1.dydx, sparse(1:numel(x1.value), 1:numel(y.value), ones(numel(x1.value),1))];
                y.pre = x1;
                x1.next = [x1.next, y];
                x1.dim = [x1.dim, numel(y.value)];
                x1.flag = [x1.flag, 0];
            else
                y = vector(x1 + x2.value);
                x2.dydx = [x2.dydx, sparse(1:numel(x1.value), 1:numel(y.value), ones(numel(x1.value),1))];
                y.pre = x2;
                x2.next = [x2.next, y];
                x2.dim = [x2.dim, numel(y.value)];
                x2.flag = [x2.flag, 0];
            end
        end
        function y = times(x1, x2)
            assert(isa(x1,'vector')|isa(x2,'vector'));
            if isa(x1,'vector')&&isa(x2,'vector')
                y = vector()
            end
            
        end
    end
end  