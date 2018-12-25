function [dydx] = back_diff(xs, y, dydy)

%%%%%%%%%%%%%%%% Lingfei Song 2018.12.13 %%%%%%%%%%%%%%
% get the derivative, reverse mode
% xs: variables against which we calculate the derivative
% y: output, class 'vector'
% dydy: initialize y.dydx, should be n*1 vector
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

y.dydx = dydy;

queue = y;

while ~isempty(queue)
    node = queue(1);
    assert(size(node.dydx, 2) == 1);    % debug;
    pre = node.pre;
    for i = 1:length(pre)
        pre_node = pre(i);
        idx = find(pre_node.next == node);
        beg = sum(pre_node.dim(1:idx-1)) + 1;
        dim = numel(node.value);
        tmp = pre_node.dydx(:,beg:beg+dim-1) * node.dydx;
        pre_node.dydx = [pre_node.dydx(:, 1:beg-1), tmp, pre_node.dydx(:,beg+dim:end)];
        pre_node.flag(idx) = 1;
        
        if isempty(find(pre_node.flag == 0, 1))
            pre_node.dydx = pre_node.dydx * ones(numel(pre_node.flag),1);
            queue = [queue, pre_node];
        end
    end
    queue(1) = [];
end

dydx = [];
for i = 1:length(xs)
    dydx = [dydx; xs(i).dydx];
end