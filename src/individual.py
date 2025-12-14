import random
import networkx as nx

class Individual:
    def __init__(self, genome, problem, dist_cache):
        self.genome = genome            
        self.problem = problem
        self.dist_cache = dist_cache    
        # path_cache rimosso: non serve durante l'evoluzione!
        self.fitness = float('inf')
        self.phenotype = []             

    def evaluate(self):
        current_node = 0
        current_weight = 0
        total_cost = 0
        
        alpha = self.problem.alpha
        beta = self.problem.beta
        
        for next_node in self.genome:
            gold = self.problem.graph.nodes[next_node]['gold']
            
            try:
                d_dir = self.dist_cache[current_node][next_node]
                d_home = self.dist_cache[current_node][0]
                d_out = self.dist_cache[0][next_node]
            except KeyError:
                self.fitness = float('inf')
                return float('inf')

            # Calcolo costi: PRIMA devo considerare se conviene tornare a casa
            # Cost per andare diretto con il peso attuale
            cost_direct = d_dir + (d_dir * alpha * current_weight) ** beta
            
            # Cost per tornare a casa e ripartire
            c_return = d_home + (d_home * alpha * current_weight) ** beta
            c_go = d_out  # peso = 0 quando riparto, quindi niente penalità
            cost_split = c_return + c_go

            if cost_split < cost_direct:
                # Torno a casa: aggiungo costo ritorno, scarico il peso, poi vado al nodo
                total_cost += c_return
                current_weight = 0
                total_cost += c_go
            else:
                # Vado diretto
                total_cost += cost_direct
            
            # IMPORTANTE: raccolgo l'oro DOPO essermi mosso (sono arrivato al nodo)
            current_weight += gold
            current_node = next_node
        
        # Ritorno finale al depot
        d_end = self.dist_cache[current_node][0]
        total_cost += d_end + (d_end * alpha * current_weight) ** beta

        self.fitness = total_cost
        return total_cost

    def rebuild_phenotype(self):
        """
        Ricostruisce il percorso come lista di tuple [(nodo, oro_preso), ...]
        Deve essere coerente con evaluate() e path_cost()
        """
        path = [(0, 0.0)]  # Inizia dal depot
        current_node = 0
        current_weight = 0
        alpha = self.problem.alpha
        beta = self.problem.beta
        graph = self.problem.graph

        for next_node in self.genome:
            gold = self.problem.graph.nodes[next_node]['gold']
            
            d_dir = self.dist_cache[current_node][next_node]
            d_home = self.dist_cache[current_node][0]
            d_out = self.dist_cache[0][next_node]

            # Stessa logica di evaluate()
            cost_direct = d_dir + (d_dir * alpha * current_weight) ** beta
            c_return = d_home + (d_home * alpha * current_weight) ** beta
            c_go = d_out
            cost_split = c_return + c_go

            if cost_split < cost_direct:
                # 1. Torna a casa (scarica)
                p_home = nx.shortest_path(graph, current_node, 0, weight='dist')
                # Aggiungi tutti i passi intermedi con 0 oro
                for step in p_home[1:]:
                    path.append((step, 0.0))
                
                # 2. Ripartire verso next_node
                p_next = nx.shortest_path(graph, 0, next_node, weight='dist')
                for step in p_next[1:-1]:  # Escludi depot e destinazione finale
                    path.append((step, 0.0))
                
                current_weight = 0
            else:
                # Vai diretto (passi intermedi)
                p_dir = nx.shortest_path(graph, current_node, next_node, weight='dist')
                for step in p_dir[1:-1]:  # Escludi nodo corrente e destinazione
                    path.append((step, 0.0))
            
            # Arrivo alla destinazione e PRENDO L'ORO
            path.append((next_node, gold))
            
            current_weight += gold
            current_node = next_node
        
        # Ritorno finale a 0
        p_end = nx.shortest_path(graph, current_node, 0, weight='dist')
        for step in p_end[1:]:
            path.append((step, 0.0))
            
        self.phenotype = path
        return path

    def mutate(self, mutation_rate=0.1):
        if random.random() < mutation_rate:
            idx1, idx2 = random.sample(range(len(self.genome)), 2)
            if idx1 > idx2:
                idx1, idx2 = idx2, idx1
            self.genome[idx1:idx2+1] = self.genome[idx1:idx2+1][::-1]