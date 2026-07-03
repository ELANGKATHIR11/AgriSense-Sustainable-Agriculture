// Custom type declarations for modules without @types packages
declare module "zustand" {
  const create: any;
  export default create;
  export const create: any;
}

declare module "@tanstack/react-query" {
  export const useQuery: any;
  export const useMutation: any;
  export const useQueryClient: any;
  export const QueryClient: any;
  export const QueryClientProvider: any;
}
