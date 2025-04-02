from itertools import zip_longest, chain

from matplotlib import pyplot as plt
from matplotlib.pyplot import savefig
import numpy as np 

from lfa_toolbox.core.fis.fis import FIS
from lfa_toolbox.core.rules.default_fuzzy_rule import DefaultFuzzyRule
from lfa_toolbox.core.rules.fuzzy_rule import FuzzyRule
from lfa_toolbox.view.mf_viewer import MembershipFunctionViewer

ANTECEDENTS_BACKGROUND_COLOR = (0.95, 0.95, 0.95)
CONSEQUENTS_BACKGROUND_COLOR = "white"


class FISViewer:
    def __init__(self, fis: FIS, figsize=None):
        self.__fis = fis
        self._has_predicted = self._get_has_predicted()

        n_default_rule = 1 if self.__fis.default_rule is not None else 0
        n_rules = len(self.__fis.rules) + n_default_rule
        max_ants = max([len(r.antecedents) for r in self.__fis.rules])
        max_cons = max([len(r.consequents) for r in self.__fis.rules])

        max_sum_ants_cons = max_ants + max_cons
        ncols = max_sum_ants_cons
        nrows = n_rules + 1  # extra row for aggregation

        if figsize is None:
            figsize = (3 * ncols, 2 * nrows)

        # Create a figure manually and add subplots so that for consequents columns
        # (columns >= max_ants) in the main block (first nrows-1 rows), the axes share
        # their x and y axes.
        fig = plt.figure(figsize=figsize)
        nrows_main = nrows - 1  # main block for rules
        axes_main = np.empty((nrows_main, ncols), dtype=object)

        for row in range(nrows_main):
            for col in range(ncols):
                subplot_index = row * ncols + col + 1  # subplot numbering starts at 1
                if row == 0 or col < max_ants:
                    # For the first row or for antecedents columns, create a normal subplot.
                    ax = fig.add_subplot(nrows, ncols, subplot_index)
                else:
                    # For consequents columns (col >= max_ants) in subsequent rows,
                    # share axes with the corresponding subplot in the first row.
                    base_ax = axes_main[0, col]
                    ax = fig.add_subplot(nrows, ncols, subplot_index,
                                         sharex=base_ax, sharey=base_ax)
                axes_main[row, col] = ax

        # Create the aggregation row (last row)
        axes_agg = []
        agg_row_index = nrows - 1
        for col in range(ncols):
            subplot_index = agg_row_index * ncols + col + 1
            ax = fig.add_subplot(nrows, ncols, subplot_index)
            axes_agg.append(ax)
        axes_agg = np.array(axes_agg).reshape(1, ncols)

        # Combine the main block and aggregation row into one array
        self._axarr = np.vstack([axes_main, axes_agg])

        if self._has_predicted:
            plt.suptitle(self._describe_fis())

        # Initially turn off all axes.
        for ax in self._axarr.flat:
            ax.axis("off")

        # Create rule plots for each rule (and the default rule, if present)
        for line, r in enumerate(chain(fis.rules, [fis.default_rule])):
            if r is not None:
                self._create_rule_plot(
                    r,
                    ax_line=self._axarr[line, :],
                    max_ants=max_ants,
                    max_cons=max_cons,
                    rule_index=line,
                )

        # Set labels for rows and columns.
        self._plot_rows_cols_labels(self._axarr, max_ants, max_cons)

        if self._has_predicted:
            # For consequents, use the aggregation row (last row)
            for cons_index, ax in enumerate(self._axarr[-1, max_ants:]):
                self._plot_aggregation(cons_index, ax)

        # If a default rule exists, show only its vertical label.
        if self.__fis.default_rule is not None:
            ax_default_rule = self._axarr[-2, 0]
            ax_default_rule.axis("on")
            ax_default_rule.set_xticks([])
            ax_default_rule.set_yticks([])
            for spine in ax_default_rule.spines.values():
                spine.set_visible(False)

    def get_axarr(self):
        return self._axarr

    @staticmethod
    def show():
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.show()

    @staticmethod
    def save(filename):
        savefig(filename, bbox_inches="tight")

    @staticmethod
    def set_title(title):
        plt.suptitle(title)

    def _create_rule_plot(self, r: FuzzyRule, ax_line, max_ants, max_cons, rule_index):
        n_rule_members = len(ax_line)

        self._plot_ants(ax_line[:max_ants], r.antecedents, n_rule_members)
        self._plot_cons(ax_line[-max_cons:], r.consequents, n_rule_members, rule_index)

    def _plot_ants(self, axarr, antecedents, n_rule_members):
        for ant, ax, i in zip_longest(
            antecedents, axarr, range(n_rule_members), fillvalue=None
        ):
            if ant is None:
                continue

            ax.axis("on")
            ax.set_facecolor(ANTECEDENTS_BACKGROUND_COLOR)

            for mf in ant.lv_name.ling_values.values():
                MembershipFunctionViewer(mf, ax=ax, color="gray", alpha=0.1)

            mf = ant.lv_name[ant.lv_value]

            not_str = "NOT " if ant.is_not else ""
            label = "[{}] {}{}".format(ant.lv_name.name, not_str, ant.lv_value)

            MembershipFunctionViewer(mf, ax=ax, label=label, draw_not=ant.is_not)

            # show last crisp inputs
            try:
                crisp_values = self.__fis.last_crisp_values
                in_value = crisp_values[ant.lv_name.name]
                fuzzified = mf.fuzzify(in_value)

                if ant.is_not:
                    fuzzified = 1.0 - fuzzified

                ax.plot([in_value], [fuzzified], "ro")
                ax.plot([in_value, in_value], [0, fuzzified], "r")
            except ValueError:
                pass

    def _plot_cons(self, axarr, consequents, n_rule_members, rule_index):
        # assumption: each rule has the same number and names of consequents
        sorted_consequents = sorted(consequents, key=lambda c: c.lv_name.name)

        for cons, ax, i in zip_longest(
            sorted_consequents, axarr, range(n_rule_members), fillvalue=None
        ):
            # print(cons, ax, i)
            if cons is None:
                continue

            ax.axis("on")
            ax.set_facecolor(CONSEQUENTS_BACKGROUND_COLOR)
            mf = cons.lv_name[cons.lv_value]
            label = "[{}] {}".format(cons.lv_name.name, cons.lv_value)
            MembershipFunctionViewer(mf, ax=ax, label=label)

            if self._has_predicted:
                mf_implicated = self.__fis.last_implicated_consequents[
                    cons.lv_name.name
                ][rule_index]
                mfv = MembershipFunctionViewer(
                    mf_implicated, ax=ax, label=label + " implicated"
                )
                mfv.fuzzify(mf_implicated.in_values[0])

    def _plot_rows_cols_labels(self, axarr, max_ants, max_cons):
        col_ants = ["Antecedent {}".format(col + 1) for col in range(max_ants)]
        col_cons = ["Consequent {}".format(col + 1) for col in range(max_cons)]

        rows = []
        for rule, row in zip(
            chain(self.__fis.rules, [self.__fis.default_rule]), range(axarr.shape[0])
        ):
            if isinstance(rule, DefaultFuzzyRule):
                rows.append("Default rule")
            elif rule is None:
                continue
            else:
                rows.append("R{} {}".format(row + 1, rule._ant_act_func[1]))

        for ax, col in zip(axarr[0], col_ants):
            ax.set_title(col)

        for ax, col in zip(axarr[0, max_ants:], col_cons):
            ax.set_title(col)

        for ax, row in zip(axarr[:, 0], rows):
            ax.set_ylabel(row, rotation=90, size="large")
            ax.yaxis.set_label_coords(-0.15, 0.5)

    def _plot_aggregation(self, cons_index, ax):
        aggr_cons = self.__fis.last_aggregated_consequents

        cons_labels = list(aggr_cons.keys())
        mf = list(aggr_cons.values())[cons_index]
        MembershipFunctionViewer(mf, ax=ax, color="orange")

        defuzz = list(self.__fis.last_defuzzified_outputs.values())[cons_index]
        ax.plot([defuzz, defuzz], [0, 1])
        ax.set_xlabel("[{}] = {:.3f}".format(cons_labels[cons_index], defuzz))

        ax.axis("on")

    def _describe_fis(self):
        if not self._has_predicted:
            return ""
        else:
            line1 = "crisp values: {}".format(self.__fis._last_crisp_values)
            line2 = "output values: {}".format(self.__fis.last_defuzzified_outputs)
            return "\n".join([line1, line2])

    def _get_has_predicted(self):
        has_pred = False
        try:
            _ = self.__fis.last_crisp_values
            has_pred = True
        except ValueError:
            pass
        return has_pred
