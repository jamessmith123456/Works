function [ hierTree ] = append_createTreeFromTable( hierTable )
% hierTable = {[2,3],[];[],[1,2];[],[3,4]};
% [ hierTree ] = append_createTreeFromTable( hierTable )

hierTree.b_isTree = 1;
hierTree = inner_createNode(hierTree, hierTable, 1);

end

function [ hierTree ] = inner_createNode(hierTree, hierTable, node_id)

hierTree.node_id = node_id;

node_link_num_child = length(hierTable{node_id,1});
node_link_num_leaf = length(hierTable{node_id,2});

node_link_num = node_link_num_child + node_link_num_leaf;
if 0 >= node_link_num
    return;
end

hierTree.branch_link = cell(node_link_num, 1);
hierTree.branch_class = cell(node_link_num, 1);
hierTree.node_classSet = [];
for i = 1:node_link_num_child
    hierTree.branch_link{i} = inner_createNode(hierTree.branch_link{i}, hierTable, hierTable{node_id,1}(i));
    hierTree.branch_class{i} = hierTree.branch_link{i}.node_classSet;
    hierTree.node_classSet = [hierTree.node_classSet, hierTree.branch_link{i}.node_classSet];
end

for i = 1:node_link_num_leaf
    hierTree.branch_class{i + node_link_num_child} = hierTable{node_id,2}(i);
end
hierTree.node_classSet = unique([hierTree.node_classSet, hierTable{node_id,2}]);

end


% hierTable_1301 = {
% [2,3],[];
% [4,5],[];
% [6,7],[];
% [8,9],[];
% [10,11],[];
% [12,13],[];
% [14,15],[];
% [16],[1];
% [17],[3];
% [18],[7];
% [19],[18];
% [20],[4];
% [21],[16];
% [22],[13];
% [],[5,6];
% [],[8,14];
% [],[21,22];
% [],[2,17];
% [],[9,15];
% [],[11,12];
% [],[20,23];
% [],[10,19];
% };


% hierTable_1302 = {
% [2,3],[];
% [4,5],[];
% [6,7],[];
% [8,9],[];
% [10,11],[];
% [12,13],[];
% [14,15],[];
% [],[1,8,14];
% [],[3,21,22];
% [],[2,7,17];
% [],[9,15,18];
% [],[4,11,12];
% [],[16,20,23];
% [],[10,13,19];
% [],[5,6];
% };
