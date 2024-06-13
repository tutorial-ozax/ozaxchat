import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { ChevronLeft, ChevronRight } from "lucide-react";
import Link from "next/link";
import { FindAllChatThreadsForReporting } from "./reporting-service";

export type ReportingProp = {
  searchParams: {
    pageSize?: number;
    pageNumber?: number;
  };
};

export const Reporting = async (props: ReportingProp) => {
  let _pageNumber = Number(props.searchParams.pageNumber ?? 0);
  let pageSize = Number(props.searchParams.pageSize ?? 10);
  let pageNumber = _pageNumber < 0 ? 0 : _pageNumber;
  let nextPage = Number(pageNumber) + 1;
  let previousPage = Number(pageNumber) - 1;

  const { resources: chatThreads } = await FindAllChatThreadsForReporting(
    pageSize,
    pageNumber
  );

  const hasMoreResults = chatThreads && chatThreads.length === pageSize;

  return (
    <Card className="h-full flex pt-8 overflow-y-auto">
      <div className="container mx-auto max-w-5xl space-y-8">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">会話履歴を表示しています。</h2>
          <p className="text-muted-foreground">全ユーザーの当月の会話履歴（管理者限定機能）</p>
        </div>
        <div className="flex items-center space-x-2">
          <Card className="flex-1">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>会話日時</TableHead>
                  <TableHead className="w-[200px]">ユーザー名</TableHead>
                  <TableHead className="mw-[300px]">タイトル</TableHead>
                  <TableHead className="w-[200px]">スレッドID</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {chatThreads &&
                  chatThreads.map((chatThread) => (
                    <TableRow key={chatThread.id}>
                      <TableCell>
                      <Link href={"/reporting/" + chatThread.id}>
                        {new Date(chatThread.createdAt).toLocaleDateString("ja-JP")}  {new Date(chatThread.createdAt).toLocaleTimeString()}
                      </Link>
                      </TableCell>
                      <TableCell>
                      <Link href={"/reporting/" + chatThread.id}>
                        {chatThread.useName}
                      </Link>
                      </TableCell>
                      <TableCell>
                      <Link href={"/reporting/" + chatThread.id}>
                        {chatThread.name}
                      </Link>
                      </TableCell>

                      <TableCell className="font-medium">
                       <Link href={"/reporting/" + chatThread.id}>
                           {chatThread.id}
                        </Link>
                      </TableCell>
                    </TableRow>
                  ))}
              </TableBody>
            </Table>
            <div className="flex gap-2 p-2 justify-end">
              {previousPage >= 0 && (
                <Button asChild size={"icon"} variant={"outline"}>
                  <Link
                    href={{
                      pathname: "/reporting",
                      search: `?pageNumber=${previousPage}`,
                    }}
                  >
                    <ChevronLeft />
                  </Link>
                </Button>
              )}
              {hasMoreResults && (
                <Button asChild size={"icon"} variant={"outline"}>
                  <Link
                    href={{
                      pathname: "/reporting",
                      search: `?pageNumber=${nextPage}`,
                    }}
                  >
                    <ChevronRight />
                  </Link>
                </Button>
              )}
            </div>
          </Card>
        </div>
      </div>
    </Card>
  );
};
