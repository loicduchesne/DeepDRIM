import random
import numpy as np
import pandas as pd
import scipy.sparse


class PBMC_CTL_DataConvert:
    def __init__(self):
        self.expr = None
        self.geneNames = None
        self.positive_pair = []  # List to hold TF-target relationships
        self.peak_set = []  # Optional: Placeholder for peak-related data if needed later

    def load_expression_data(self, expr_file):
        """Loads the expression data for PBMC-CTL dataset."""
        df = pd.read_csv(expr_file, index_col=0)
        self.expr = scipy.sparse.csr_matrix(df.values)  # Sparse matrix for memory efficiency
        self.geneNames = np.array(df.index)
        print(f"Expression data loaded: {self.expr.shape[0]} genes, {self.expr.shape[1]} cells.")

    def load_imposed_grn(self, grn_file):
        """Loads the imposed GRN file to extract TF-target relationships."""
        grn_data = pd.read_csv(grn_file)

        # Debugging: Print column names if column mismatch occurs
        print("Columns in GRN file:", grn_data.columns)

        # Adjust column names as per the dataset
        tf_column = "TF (regulator)"  # Replace with actual column name for TFs
        target_column = "Gene (target)"  # Replace with actual column name for targets

        if tf_column not in grn_data.columns or target_column not in grn_data.columns:
            raise ValueError(f"Columns '{tf_column}' or '{target_column}' not found in {grn_file}.")

        for _, row in grn_data.iterrows():
            tf, target = row[tf_column], row[target_column]
            self.positive_pair.append(f"{tf},{target}")
        print(f"Loaded {len(self.positive_pair)} TF-target pairs from GRN.")

    def output_positive_pair_set(self, outfile_positive_pairs):
        """Outputs the positive pairs in the required format."""
        np.savetxt(outfile_positive_pairs, self.positive_pair, delimiter='\n', fmt='%s')
        print(f"Positive pairs saved to {outfile_positive_pairs}.")

    def output_gene_names(self, outfile_gene_names):
        """Outputs gene names if needed for downstream processes."""
        np.savetxt(outfile_gene_names, self.geneNames, delimiter='\n', fmt='%s')
        print(f"Gene names saved to {outfile_gene_names}.")

    def output_geneName_map(self, outfile_geneName_map):
        """Outputs a gene name map file."""
        if self.geneNames is not None:
            gene_map = pd.DataFrame({
                'geneName': self.geneNames,
                'mappedGeneName': self.geneNames
            })
            gene_map.to_csv(outfile_geneName_map, sep="\t", index=False, header=False)
            print(f"Gene name map saved to {outfile_geneName_map}.")
        else:
            print("Gene names are not loaded. Cannot generate geneName_map.txt.")

    def generate_training_pairs(self, output_file, negative_sample_ratio=1.0, randomize=False):
        """Generates training pairs with positive and placeholder negative examples."""

        # Positive pairs (1)
        positive_pairs = pd.DataFrame(
            [pair.split(",") + [1] for pair in self.positive_pair],
            columns=["GeneA", "GeneB", "Label"]
        )

        # Reverse pairs (2)
        reverse_pairs = pd.DataFrame({
            "GeneA": positive_pairs["GeneB"],
            "GeneB": positive_pairs["GeneA"],
            "Label": [2] * len(positive_pairs),
        })

        # Combine positive and reverse pairs
        all_pairs = pd.concat([positive_pairs, reverse_pairs])

        # Negative pairs (0)
        gene_list = list(self.geneNames)
        num_negative_samples = int(len(all_pairs) * negative_sample_ratio)

        negative_pairs = []
        for _ in range(num_negative_samples):
            gene_a, gene_b = random.sample(gene_list, 2)
            if not ((gene_a, gene_b) in all_pairs.values or (gene_b, gene_a) in all_pairs.values):
                negative_pairs.append((gene_a, gene_b, 0))

        negative_pairs_df = pd.DataFrame(negative_pairs, columns=["GeneA", "GeneB", "Label"])

        # Combine all
        training_data = pd.concat([all_pairs, negative_pairs_df])

        # Shuffle
        if randomize:
            training_data = training_data.sample(frac=1).reset_index(drop=True)

        training_data.to_csv(output_file, sep="\t", index=False, header=False)
        print(f"Training pairs saved to {output_file}.")

    def work_pbmc_ctl_to_positive_pairs(self, expr_file, grn_file, output_prefix):
        """Combines all steps to process the PBMC-CTL data."""
        self.load_expression_data(expr_file)
        self.load_imposed_grn(grn_file)
        self.output_positive_pair_set(f"{output_prefix}_positive_pairs.csv")
        self.generate_training_pairs(f"{output_prefix}_training_pairs.txt")
        self.output_geneName_map(f"{output_prefix}_geneName_map.txt")
        self.output_gene_names(f"{output_prefix}_gene_names.txt")


if __name__ == "__main__":
    # Initialize and run the conversion
    converter = PBMC_CTL_DataConvert()
    converter.work_pbmc_ctl_to_positive_pairs(
        expr_file="data/PBMC-CTL_1000_cells.csv",
        grn_file="data/PBMC-CTL_Imposed_GRN.csv",
        output_prefix="out/DeepDRIM/processed/PBMC-CTL"
    )
