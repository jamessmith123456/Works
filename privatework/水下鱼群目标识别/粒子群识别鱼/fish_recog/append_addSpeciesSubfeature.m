function [ hierTree ] = append_addSpeciesSubfeature( hierTree )
%APPEND_ADDSPECIESSUBFEATURE Summary of this function goes here

hierTree.SpeciesSubfeature = inner_addSpeciesSubfeature(hierTree, {});


end

function subfeatureCell = inner_addSpeciesSubfeature(hierTree, subfeatureCell)

if isempty(hierTree)
    return;
end

for i = 1:length(hierTree.branch_class)
    if length(hierTree.branch_class{i}) == 1
        subfeatureCell{hierTree.branch_class{i}} = hierTree.Subfeature;
    else
        subfeatureCell = inner_addSpeciesSubfeature(hierTree.branch_link{i}, subfeatureCell);
    end
end

end
