import React, { useState, useMemo, useEffect } from 'react';
import { cn } from '@/utils/cn';
import { ChevronDown, ChevronUp, SortAsc, SortDesc, Search } from 'lucide-react';

export interface Column<T> {
  key: string;
  header: string;
  render?: (item: T) => React.ReactNode;
  sortable?: boolean;
  className?: string;
  cellClassName?: string;
}

export interface DataTableProps<T> {
  data: T[];
  columns: Column<T>[];
  keyExtractor: (item: T) => string;
  isLoading?: boolean;
  className?: string;
  emptyMessage?: string;
  pagination?: boolean;
  pageSize?: number;
  searchable?: boolean;
  searchPlaceholder?: string;
}

function DataTable<T>({
  data,
  columns,
  keyExtractor,
  isLoading = false,
  className,
  emptyMessage = 'No data available',
  pagination = false,
  pageSize = 10,
  searchable = false,
  searchPlaceholder = 'Search...'
}: DataTableProps<T>) {
  const [sortConfig, setSortConfig] = useState<{
    key: string;
    direction: 'ascending' | 'descending';
  } | null>(null);
  
  const [currentPage, setCurrentPage] = useState(1);
  const [searchTerm, setSearchTerm] = useState('');
  
  // Reset to first page when data changes
  useEffect(() => {
    setCurrentPage(1);
  }, [data]);
  
  // Handle sorting
  const sortedData = useMemo(() => {
    let sortableData = [...data];
    
    // Apply search if enabled
    if (searchable && searchTerm) {
      sortableData = sortableData.filter(item => {
        return Object.values(item).some(value => 
          String(value).toLowerCase().includes(searchTerm.toLowerCase())
        );
      });
    }
    
    // Apply sorting if configured
    if (sortConfig !== null) {
      sortableData.sort((a, b) => {
        const aValue = (a as any)[sortConfig.key];
        const bValue = (b as any)[sortConfig.key];
        
        if (aValue === undefined || bValue === undefined) return 0;
        
        if (aValue < bValue) {
          return sortConfig.direction === 'ascending' ? -1 : 1;
        }
        if (aValue > bValue) {
          return sortConfig.direction === 'ascending' ? 1 : -1;
        }
        return 0;
      });
    }
    
    return sortableData;
  }, [data, sortConfig, searchTerm, searchable]);
  
  // Handle pagination
  const paginatedData = useMemo(() => {
    if (!pagination) return sortedData;
    
    const startIndex = (currentPage - 1) * pageSize;
    return sortedData.slice(startIndex, startIndex + pageSize);
  }, [sortedData, currentPage, pageSize, pagination]);
  
  const pageCount = useMemo(() => {
    return Math.ceil(sortedData.length / pageSize);
  }, [sortedData, pageSize]);
  
  const requestSort = (key: string) => {
    let direction: 'ascending' | 'descending' = 'ascending';
    
    if (sortConfig && sortConfig.key === key && sortConfig.direction === 'ascending') {
      direction = 'descending';
    }
    
    setSortConfig({ key, direction });
  };
  
  return (
    <div className={cn("flex flex-col", className)}>
      {/* Search bar */}
      {searchable && (
        <div className="flex items-center mb-4 w-full max-w-md">
          <div className="relative w-full">
            <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none text-muted-foreground">
              <Search size={16} />
            </div>
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder={searchPlaceholder}
              className="bg-background border border-border text-foreground text-sm rounded-md focus:ring-primary focus:border-primary block w-full pl-10 p-2.5"
            />
          </div>
        </div>
      )}
      
      {/* Table */}
      <div className="relative overflow-x-auto rounded-md border border-border">
        <table className="w-full text-sm">
          <thead className="bg-muted/30 text-left">
            <tr>
              {columns.map((column) => (
                <th 
                  key={column.key} 
                  className={cn(
                    "px-4 py-3 font-medium text-muted-foreground whitespace-nowrap",
                    column.sortable && "cursor-pointer select-none",
                    column.className
                  )}
                  onClick={column.sortable ? () => requestSort(column.key) : undefined}
                >
                  <div className="flex items-center gap-2">
                    {column.header}
                    {column.sortable && (
                      <div className="flex flex-col">
                        {sortConfig?.key === column.key ? (
                          sortConfig.direction === 'ascending' ? 
                            <SortAsc size={14} className="text-primary" /> : 
                            <SortDesc size={14} className="text-primary" />
                        ) : (
                          <div className="flex flex-col">
                            <ChevronUp size={10} className="text-muted-foreground" />
                            <ChevronDown size={10} className="text-muted-foreground -mt-1" />
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {isLoading ? (
              <tr>
                <td colSpan={columns.length} className="px-4 py-8 text-center text-muted-foreground">
                  Loading...
                </td>
              </tr>
            ) : paginatedData.length > 0 ? (
              paginatedData.map((item) => (
                <tr key={keyExtractor(item)} className="border-t border-border hover:bg-muted/20">
                  {columns.map((column) => (
                    <td 
                      key={`${keyExtractor(item)}-${column.key}`} 
                      className={cn("px-4 py-3", column.cellClassName)}
                    >
                      {column.render ? column.render(item) : (item as any)[column.key]}
                    </td>
                  ))}
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={columns.length} className="px-4 py-8 text-center text-muted-foreground">
                  {emptyMessage}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      
      {/* Pagination */}
      {pagination && pageCount > 1 && (
        <div className="flex items-center justify-between mt-4">
          <div className="text-sm text-muted-foreground">
            Showing {((currentPage - 1) * pageSize) + 1} - {Math.min(currentPage * pageSize, sortedData.length)} of {sortedData.length} items
          </div>
          <div className="flex space-x-1">
            <button
              onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
              disabled={currentPage === 1}
              className={cn(
                "px-3 py-1 rounded-md text-sm",
                currentPage === 1 
                  ? "text-muted-foreground bg-muted cursor-not-allowed"
                  : "text-foreground bg-muted hover:bg-muted/80"
              )}
            >
              Previous
            </button>
            
            {Array.from({ length: Math.min(5, pageCount) }, (_, i) => {
              // Show pages around current page
              let pageNum;
              if (pageCount <= 5) {
                pageNum = i + 1;
              } else if (currentPage <= 3) {
                pageNum = i + 1;
              } else if (currentPage >= pageCount - 2) {
                pageNum = pageCount - 4 + i;
              } else {
                pageNum = currentPage - 2 + i;
              }
              
              return (
                <button
                  key={pageNum}
                  onClick={() => setCurrentPage(pageNum)}
                  className={cn(
                    "px-3 py-1 rounded-md text-sm",
                    currentPage === pageNum 
                      ? "bg-primary text-primary-foreground"
                      : "text-foreground bg-muted hover:bg-muted/80"
                  )}
                >
                  {pageNum}
                </button>
              );
            })}
            
            <button
              onClick={() => setCurrentPage(p => Math.min(pageCount, p + 1))}
              disabled={currentPage === pageCount}
              className={cn(
                "px-3 py-1 rounded-md text-sm",
                currentPage === pageCount 
                  ? "text-muted-foreground bg-muted cursor-not-allowed"
                  : "text-foreground bg-muted hover:bg-muted/80"
              )}
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default DataTable;
